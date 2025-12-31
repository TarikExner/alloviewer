// === Batch mask + COM + per-ROI channel means (recursive) ===
// Outputs per image:
//   1) <base>_mask.tif
//   2) <base>_data.csv  columns: X,Y,mean_red,mean_green,mean_blue
// Master at root:
//   master_stats.csv    columns: image_path,image_name,n_cells

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

// Master CSV at root (only n_cells)
masterPath = root + "master_stats.csv";
if (!File.exists(masterPath)) {
    File.saveString("image_path,image_name,n_cells\n", masterPath);
}

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

    open(dir + name);
    origTitle = getTitle();
    getDimensions(width, height, channels, slices, frames);

    base = stripExt(name);
    maskPath    = dir + base + "_mask.tif";
    dataCsvPath = dir + base + "_data.csv"; // X,Y,mean_red,mean_green,mean_blue

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

    // If we can't segment (need at least red+green), write empty outputs and exit
    if (redTitle == "" || greenTitle == "") {
        saveEmptyMaskAndCSV(width, height, maskPath, dataCsvPath);
        appendMaster(dir + name, base, 0);
        closeAllLike(origTitle);
        return;
    }

    // Make read-only duplicates for measurement (like your second macro)
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
        if (filteredMaskTitle != "") { safeClose(filteredMaskTitle); }
        saveEmptyMaskAndCSV(width, height, maskPath, dataCsvPath);
        appendMaster(dir + name, base, 0);

        // cleanup
        safeClose(segTitle);
        safeClose(readRedTitle);
        safeClose(readGreenTitle);
        safeClose(readBlueTitle);
        closeAllLike(origTitle);
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
        run("Measure"); // XM, YM (also Mean, but we ignore it here)
    }
    comX = newArray(n); comY = newArray(n);
    for (r = 0; r < n; r++) {
        comX[r] = getResult("XM", r);
        comY[r] = getResult("YM", r);
    }

    // -------- Mean intensities on RED, GREEN, BLUE (read-only duplicates) --------
    // RED
    redMeans = newArray(n);
    run("Clear Results");
    selectWindow(readRedTitle);
    for (r = 0; r < n; r++) {
        roiManager("Select", r);
        run("Measure");
    }
    for (r = 0; r < n; r++) redMeans[r] = getResult("Mean", r);

    // GREEN
    greenMeans = newArray(n);
    run("Clear Results");
    selectWindow(readGreenTitle);
    for (r = 0; r < n; r++) {
        roiManager("Select", r);
        run("Measure");
    }
    for (r = 0; r < n; r++) greenMeans[r] = getResult("Mean", r);

    // BLUE (if missing, fill with -1)
    blueMeans = newArray(n);
    if (readBlueTitle != "") {
        run("Clear Results");
        selectWindow(readBlueTitle);
        for (r = 0; r < n; r++) {
            roiManager("Select", r);
            run("Measure");
        }
        for (r = 0; r < n; r++) blueMeans[r] = getResult("Mean", r);
    } else {
        for (r = 0; r < n; r++) blueMeans[r] = -1;
    }
    
    folderName = stripTrailingSlash(dir);
	folderName = basename(folderName);

    // -------- SAVE per-image CSV: X,Y,mean_red,mean_green,mean_blue --------
    saveDataCsv(dataCsvPath, folderName, name, comX, comY, redMeans, greenMeans, blueMeans);

    // -------- Master stats (only n_cells) --------
    appendMaster(dir + name, base, n);

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

function appendMaster(imagePath, imageName, nCells) {
    File.append(imagePath + "," + imageName + "," + nCells + "\n", masterPath);
}

// ----- Utilities -----
function ensureRoiManager() {
    if (!isOpen("ROI Manager")) run("ROI Manager...");
}

// Try to find a window belonging to this image that matches a channel tag
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

// Close all windows related to a given base title
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

// Return the newly created window title after an operation
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

// Drop extension from filename
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

function d2(x) { return toString(round(x*100)/100); }
function d3(x) { return toString(round(x*1000)/1000); }
