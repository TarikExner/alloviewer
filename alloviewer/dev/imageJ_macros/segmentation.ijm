// === Batch mask + COM + per-ROI channel means (recursive) ===
// Masks: saved into sibling folder "<folder>_masks" next to each image folder
// CSVs : saved into ROOT/results/ with ascending 5-digit IDs (00001.csv, ...)
// Per-image CSV columns: Folder,file_name,X,Y,mean_red,mean_green,mean_blue
// Master CSV at root: master_stats.csv with only n_cells

// ------ CONFIG ------
minSize = 10;    // px
maxSize = 50;    // px
minCirc = 0.80;
maxCirc = 1.00;

// ------ CHOOSE ROOT FOLDER ------
root = getDirectory("Choose the ROOT folder (will recurse into subfolders)");
rstr = "" + root;
if (rstr == "null" || rstr == "") {
    showMessage("No folder selected.");
    exit();
}

sep = pathSep(root);

// Global results folder at root
resultsDir = root + "results" + sep;
if (!File.isDirectory(resultsDir)) File.makeDirectory(resultsDir);

// Master CSV at root (only n_cells)
masterPath = root + "master_stats.csv";
if (!File.exists(masterPath)) {
    File.saveString("n_cells\n", masterPath);
}

// Global CSV counter (ascending across the whole run)
csvCounter = 1;

setBatchMode(true);

// Global options that don't require an open image
run("Options...", "iterations=1 count=1 black do=Nothing");   // binary: background black
run("Set Measurements...", "mean center of mass redirect=None decimal=3"); // Mean + XM,YM

processDir(root);

setBatchMode(false);
print("Done.");


// ----------------- FUNCTIONS -----------------

function processDir(dir) {
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        path = dir + list[i];
        if (File.isDirectory(path)) {
            processDir(path);
        } else {
            lower = toLowerCase(path);
            if (endsWith(lower, ".tif") || endsWith(lower, ".tiff") ||
                endsWith(lower, ".png") || endsWith(lower, ".jpg") ||
                endsWith(lower, ".jpeg") || endsWith(lower, ".bmp")) {
                processImage(dir, list[i]);
            }
        }
    }
}

function processImage(dir, name) {
    ensureRoiManager();
    roiManager("Reset");
    run("Clear Results");

    // CSV id and path (global ascending)
    csvId = pad5(csvCounter);
    dataCsvPath = resultsDir + csvId + ".csv";

    // Folder name only (last folder in path)
    folderName = basename(stripTrailingSlash(dir));

    // Masks folder: sibling "<dir_without_slash>_masks/"
	localSep = pathSep(dir);
	
	dirNoSlash = dir;
	if (endsWith(dirNoSlash, "/") || endsWith(dirNoSlash, "\\")) {
	    dirNoSlash = substring(dirNoSlash, 0, lengthOf(dirNoSlash)-1);
	}
	
	masksDir = dirNoSlash + "_masks" + localSep;
    if (!File.isDirectory(masksDir)) File.makeDirectory(masksDir);

    open(dir + name);
    origTitle = getTitle();
    getDimensions(width, height, channels, slices, frames);

    base = stripExt(name);
    maskPath = masksDir + base + "_mask.tif";

    // Set Scale AFTER opening the image (avoids error)
    run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");

    // Split channels
    run("Split Channels");

    redTitle   = findChannelWindow(origTitle, "red");
    greenTitle = findChannelWindow(origTitle, "green");
    blueTitle  = findChannelWindow(origTitle, "blue");

    if (redTitle == "")   redTitle   = findChannelWindow(origTitle, "c1");
    if (greenTitle == "") greenTitle = findChannelWindow(origTitle, "c2");
    if (blueTitle == "")  blueTitle  = findChannelWindow(origTitle, "c3");

    // Need at least red+green for segmentation
    if (redTitle == "" || greenTitle == "") {
        saveEmptyMaskAndCSV(width, height, maskPath, dataCsvPath);
        appendMaster(0);

        closeAllLike(origTitle);

        csvCounter++; // still consume an ID
        return;
    }

    // Make read-only duplicates for measurement
    selectWindow(redTitle);
    run("Duplicate...", "title=" + base + "-red-read");
    readRedTitle = getTitle();

    selectWindow(greenTitle);
    run("Duplicate...", "title=" + base + "-green-read");
    readGreenTitle = getTitle();

    readBlueTitle = "";
    if (blueTitle != "") {
        selectWindow(blueTitle);
        run("Duplicate...", "title=" + base + "-blue-read");
        readBlueTitle = getTitle();
    }

    // -------- SEGMENTATION: (Red + Green) -> 8-bit -> BG subtract -> Phansalkar -> Watershed --------
    run("Image Calculator...", "operation=Add image1=["+redTitle+"] image2=["+greenTitle+"] create");
    segTitle = getTitle();

    selectWindow(segTitle);
    run("8-bit");
    run("Subtract Background...", "rolling=20 sliding paraboloid");
    run("Subtract...", "value=40");
    run("Auto Local Threshold", "method=Phansalkar radius=2 parameter_1=0 parameter_2=0 white");
    run("Watershed");

    // -------- ROIs + FILTERED MASK from Analyze Particles --------
    ensureRoiManager();
    roiManager("Reset");

    titlesBefore = getList("image.titles");

    run("Analyze Particles...",
        "size=" + minSize + "-" + maxSize +
        " circularity=" + d2(minCirc) + "-" + d2(maxCirc) +
        " add show=Masks pixel");

    titlesAfter = getList("image.titles");
    filteredMaskTitle = findNewWindow(titlesBefore, titlesAfter);

    n = roiManager("count");

    if (n == 0) {
        if (filteredMaskTitle != "") safeClose(filteredMaskTitle);

        // still save outputs
        saveEmptyMaskAndCSV(width, height, maskPath, dataCsvPath);
        appendMaster(0);

        // cleanup
        safeClose(segTitle);
        safeClose(readRedTitle);
        safeClose(readGreenTitle);
        safeClose(readBlueTitle);
        closeAllLike(origTitle);

        csvCounter++;
        return;
    }

    // save the filtered mask (true binary 0/255)
    if (filteredMaskTitle != "") {
        selectWindow(filteredMaskTitle);
        run("8-bit");
        setThreshold(1, 255);
        run("Convert to Mask");
        run("Rename...", base + "_mask");
        saveAs("Tiff", maskPath);
        close();
    }

    // -------- COM on the segmented image (XM, YM) --------
    run("Clear Results");
    selectWindow(segTitle);
    for (r = 0; r < n; r++) {
        roiManager("Select", r);
        run("Measure");
    }
    comX = newArray(n); comY = newArray(n);
    for (r = 0; r < n; r++) {
        comX[r] = getResult("XM", r);
        comY[r] = getResult("YM", r);
    }

    // -------- Mean intensities on read duplicates --------
    // RED
    redMeans = newArray(n);
    run("Clear Results");
    selectWindow(readRedTitle);
    for (r = 0; r < n; r++) { roiManager("Select", r); run("Measure"); }
    for (r = 0; r < n; r++) redMeans[r] = getResult("Mean", r);

    // GREEN
    greenMeans = newArray(n);
    run("Clear Results");
    selectWindow(readGreenTitle);
    for (r = 0; r < n; r++) { roiManager("Select", r); run("Measure"); }
    for (r = 0; r < n; r++) greenMeans[r] = getResult("Mean", r);

    // BLUE (if missing, fill with -1)
    blueMeans = newArray(n);
    if (readBlueTitle != "") {
        run("Clear Results");
        selectWindow(readBlueTitle);
        for (r = 0; r < n; r++) { roiManager("Select", r); run("Measure"); }
        for (r = 0; r < n; r++) blueMeans[r] = getResult("Mean", r);
    } else {
        for (r = 0; r < n; r++) blueMeans[r] = -1;
    }

    // -------- SAVE per-image CSV into ROOT/results/ as 5-digit ID --------
    saveDataCsv(dataCsvPath, folderName, name, comX, comY, redMeans, greenMeans, blueMeans);

    // -------- Master stats (only n_cells) --------
    appendMaster(n);

    // -------- CLEANUP --------
    safeClose(segTitle);
    safeClose(readRedTitle);
    safeClose(readGreenTitle);
    safeClose(readBlueTitle);
    closeAllLike(origTitle);

    ensureRoiManager();
    roiManager("Reset");
    run("Clear Results");
    run("Collect Garbage");
    call("java.lang.System.gc");

    csvCounter++;
}

// ----- Save helpers -----
function saveEmptyMaskAndCSV(w, h, maskPath, dataCsvPath) {
    // empty mask
    setForegroundColor(255,255,255);
    setBackgroundColor(0,0,0);
    newImage("tmp_mask", "8-bit black", w, h, 1);
    setThreshold(1, 255); run("Convert to Mask");
    saveAs("Tiff", maskPath);
    close();

    // empty CSV
    saveEmptyCSV(dataCsvPath);
}

function saveEmptyCSV(path) {
    File.saveString("Folder,file_name,X,Y,mean_red,mean_green,mean_blue\n", path);
}

function saveDataCsv(path, folder, fileName, xs, ys, mR, mG, mB) {
    txt = "Folder,file_name,X,Y,mean_red,mean_green,mean_blue\n";
    for (i=0; i<xs.length; i++) {
        txt = txt + folder + "," + fileName + "," +
              d3(xs[i]) + "," + d3(ys[i]) + "," +
              d3(mR[i]) + "," + d3(mG[i]) + "," + d3(mB[i]) + "\n";
    }
    File.saveString(txt, path);
}

function appendMaster(nCells) {
    File.append(nCells + "\n", masterPath);
}

// ----- Utilities -----
function ensureRoiManager() {
    if (!isOpen("ROI Manager")) run("ROI Manager...");
}

function findChannelWindow(base, tag) {
    possibilities = newArray(
        base + " (" + tag + ")",
        base + "-" + tag,
        base + " " + tag,
        tag
    );
    for (ii=0; ii<possibilities.length; ii++) {
        t = possibilities[ii];
        if (isOpen(t)) return t;
    }
    titles = getList("image.titles");
    for (ii=0; ii<titles.length; ii++) {
        t = titles[ii];
        if (indexOf(toLowerCase(t), toLowerCase(base))>=0 && indexOf(toLowerCase(t), toLowerCase(tag))>=0)
            return t;
    }
    return "";
}

function closeAllLike(base) {
    titles = getList("image.titles");
    baseLower  = toLowerCase(base);
    baseStem   = toLowerCase(stripExt(base));

    for (ii = titles.length - 1; ii >= 0; ii--) {
        t = titles[ii];
        tl = toLowerCase(t);
        tStem = toLowerCase(stripExt(t));

        if (
            tl == baseLower ||
            tStem == baseStem ||
            startsWith(tl, baseLower) ||
            startsWith(tl, baseStem) ||
            indexOf(tl, baseStem + " (") >= 0 ||
            indexOf(tl, baseStem + "_") >= 0 ||
            indexOf(tl, baseStem + "-") >= 0 ||
            indexOf(toLowerCase(t), toLowerCase(base)) >= 0
        ) {
            selectWindow(t);
            close();
        }
    }
}

function findNewWindow(before, after) {
    for (i = 0; i < after.length; i++) {
        exists = false;
        for (j = 0; j < before.length; j++) {
            if (after[i] == before[j]) { exists = true; break; }
        }
        if (!exists) return after[i];
    }
    return "";
}

function stripExt(f) {
    dot = lastIndexOf(f, ".");
    if (dot < 0) return f;
    return substring(f, 0, dot);
}

function safeClose(title) {
    if (title != "" && isOpen(title)) {
        selectWindow(title);
        close();
    }
}

function stripTrailingSlash(p) {
    if (endsWith(p, "/") || endsWith(p, "\\")) return substring(p, 0, lengthOf(p)-1);
    return p;
}

function basename(p) {
    p = replace(p, "\\", "/");
    last = lastIndexOf(p, "/");
    if (last < 0) return p;
    return substring(p, last+1);
}

// Decide which separator is used in a path string
function pathSep(p) {
    if (indexOf(p, "\\") >= 0) return "\\";
    return "/";
}

function pad5(n) {
    s = "" + n;
    while (lengthOf(s) < 5) s = "0" + s;
    return s;
}

function d2(x) { return toString(round(x*100)/100); }
function d3(x) { return toString(round(x*1000)/1000); }
