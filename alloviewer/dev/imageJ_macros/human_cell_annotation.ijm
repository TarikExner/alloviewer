// === CONFIG ===
minSize = 10;    // px
maxSize = 50;    // px
minCirc = 0.80;
maxCirc = 1.00;

// thresholds kept in case they're useful later
maxRedCut   = 1.1;
minGreenCut = 1.11;

var root, mappingPath, outCsvPath;

var mapFolder, mapOrientation, mapType, mapPRA, mapSpec, mapWell;
var mapAnnot1, mapAnnot2, mapImage, mapCount;

var doneKey, doneCount;

var segCount, greenCount, redCount;

// already processed (for resume)
doneKey   = newArray(0);
doneCount = 0;

// global ROI counters
segCount   = 0;
greenCount = 0;
redCount   = 0;

var pendingLines;

// ------------- MAIN -------------

macro "BatchSegWithUserDots_FromCSV" {

    // ------ ROOT FOLDER ------
    root = getDirectory("Choose the ROOT folder (parent of all mapping folders)");
    rstr = "" + root;
    if (rstr == "null" || rstr == "") {
        showMessage("No folder selected.");
        exit();
    }

    // ------ MAPPING CSV ------
    mappingPath = File.openDialog("Select mapping CSV");
    mstr = "" + mappingPath;
    if (mstr == "null" || mstr == "") {
        showMessage("No mapping CSV selected.");
        exit();
    }

    // Load mapping from CSV with all columns
    loadMapping(mappingPath);
    if (mapCount == 0) {
        showMessage("Mapping CSV seems empty or has no valid lines.");
        exit();
    }

	// Output CSV
	outCsvPath = root + "roi_stats.csv";
	if (!File.exists(outCsvPath)) {
	    // slimmer header: keep Folder, image_name, roi_id, roi_type + means
	    header = "Folder,image_name,roi_id,roi_type,mean_red,mean_green,mean_blue\n";
	    File.saveString(header, outCsvPath);
	}

    // Load done set to support resume (based on Folder + image_name)
    loadDoneSet(outCsvPath);

    setBatchMode(true);

    // we only need mean, no COM
    run("Options...", "iterations=1 count=1 black do=Nothing");
    run("Set Measurements...", "mean redirect=None decimal=3");

    // Loop over mapping rows
    for (i = 0; i < mapCount; i++) {
        folder    = mapFolder[i];
        orient    = mapOrientation[i];
        type      = mapType[i];
        pra       = mapPRA[i];
        spec      = mapSpec[i];
        well      = mapWell[i];
        annot1    = mapAnnot1[i];
        annot2    = mapAnnot2[i];
        imageName = mapImage[i];

        if (isDone(folder, imageName)) {
            print("Skipping already processed: " + folder + " / " + imageName);
            continue;
        }

        // Build dir path: root + folder + separator
        dir = root + folder;
        dir = dir + File.separator;

        fullPath = dir + imageName;
        if (!File.exists(fullPath)) {
            print("File not found, skipping: " + fullPath);
            continue;
        }

        print("Processing: " + folder + " / " + imageName);
        processImage(dir, imageName,
                     folder, orient, type, pra, spec, well, annot1, annot2);
    }

    setBatchMode(false);
    print("Done.");
}

// ----------------- LOAD MAPPING -----------------

function loadMapping(path) {
    txt = File.openAsString(path);

    // normalise line endings
    txt = replace(txt, "\r", "");

    lines = split(txt, "\n");
    if (lines.length <= 1) {
        mapCount = 0;
        return;
    }

    // --- header ---
    header = lines[0];
    header = trim(header);

    // detect separator: if header contains ';', use that, else comma
    sep = ",";
    if (indexOf(header, ";") >= 0) {
        sep = ";";
    }

    headerCols = split(header, sep);

    idxFolder = findColIndex(headerCols, "Folder");
    idxOrient = findColIndex(headerCols, "Orientation");
    idxType   = findColIndex(headerCols, "Type");
    idxPRA    = findColIndex(headerCols, "PRA");
    idxSpec   = findColIndex(headerCols, "Specificity");
    idxWell   = findColIndex(headerCols, "Well");
    idxAnnot1 = findColIndex(headerCols, "annotator_1");
    idxAnnot2 = findColIndex(headerCols, "annotator_2");
    idxImage  = findColIndex(headerCols, "image_name");

    if (idxFolder < 0 || idxImage < 0 || idxWell < 0) {
        showMessage("CSV is missing required columns (Folder, Well, image_name).");
        mapCount = 0;
        return;
    }

    maxN = lines.length - 1; // skip header
    mapFolder      = newArray(maxN);
    mapOrientation = newArray(maxN);
    mapType        = newArray(maxN);
    mapPRA         = newArray(maxN);
    mapSpec        = newArray(maxN);
    mapWell        = newArray(maxN);
    mapAnnot1      = newArray(maxN);
    mapAnnot2      = newArray(maxN);
    mapImage       = newArray(maxN);
    mapCount       = 0;
    

    for (i = 1; i < lines.length; i++) {
        line = lines[i];
        line = trim(line);
        if (line == "") {
            continue;
        }

        cols = split(line, sep);
        if (cols.length <= idxImage) {
            continue;
        }

        // Folder
        if (idxFolder >= 0 && idxFolder < cols.length) {
            mapFolder[mapCount] = trim(cols[idxFolder]);
        } else {
            mapFolder[mapCount] = "";
        }

        // Orientation
        if (idxOrient >= 0 && idxOrient < cols.length) {
            mapOrientation[mapCount] = trim(cols[idxOrient]);
        } else {
            mapOrientation[mapCount] = "";
        }

        // Type
        if (idxType >= 0 && idxType < cols.length) {
            mapType[mapCount] = trim(cols[idxType]);
        } else {
            mapType[mapCount] = "";
        }

        // PRA
        if (idxPRA >= 0 && idxPRA < cols.length) {
            mapPRA[mapCount] = trim(cols[idxPRA]);
        } else {
            mapPRA[mapCount] = "";
        }

        // Specificity
        if (idxSpec >= 0 && idxSpec < cols.length) {
            mapSpec[mapCount] = trim(cols[idxSpec]);
        } else {
            mapSpec[mapCount] = "";
        }

        // Well
        if (idxWell >= 0 && idxWell < cols.length) {
            mapWell[mapCount] = trim(cols[idxWell]);
        } else {
            mapWell[mapCount] = "";
        }

        // annotator_1
        if (idxAnnot1 >= 0 && idxAnnot1 < cols.length) {
            mapAnnot1[mapCount] = trim(cols[idxAnnot1]);
        } else {
            mapAnnot1[mapCount] = "";
        }

        // annotator_2
        if (idxAnnot2 >= 0 && idxAnnot2 < cols.length) {
            mapAnnot2[mapCount] = trim(cols[idxAnnot2]);
        } else {
            mapAnnot2[mapCount] = "";
        }

        // image_name
        if (idxImage >= 0 && idxImage < cols.length) {
            mapImage[mapCount] = trim(cols[idxImage]);
        } else {
            mapImage[mapCount] = "";
        }

        mapCount++;
    }
}

function findColIndex(cols, target) {
    for (i = 0; i < cols.length; i++) {
        if (trim(cols[i]) == target) {
            return i;
        }
    }
    return -1;
}

// ----------------- LOAD DONE SET (RESUME) -----------------

function loadDoneSet(path) {
    doneCount = 0;

    txt = File.openAsString(path);
    txt = replace(txt, "\r", "");

    lines = split(txt, "\n");
    if (lines.length <= 1) {
        return; // only header
    }

    header = lines[0];
    header = trim(header);
    hcols  = split(header, ",");

    idxFolder = findColIndex(hcols, "Folder");
    idxImage  = findColIndex(hcols, "image_name");

    if (idxFolder < 0 || idxImage < 0) {
        return;
    }

    doneKey = newArray(lines.length - 1);

    for (i = 1; i < lines.length; i++) {
        line = lines[i];
        line = trim(line);
        if (line == "") {
            continue;
        }

        cols = split(line, ",");
        if (cols.length <= idxImage) {
            continue;
        }

        folder = trim(cols[idxFolder]);
        image  = trim(cols[idxImage]);

        key = folder + "|" + image;
        doneKey[doneCount] = key;
        doneCount++;
    }
}

function isDone(folder, imageName) {
    key = folder + "|" + imageName;
    for (i = 0; i < doneCount; i++) {
        if (doneKey[i] == key) {
            return true;
        }
    }
    return false;
}

// ----------------- PER-IMAGE PROCESSING -----------------

function processImage(dir, name,
                      folderKey, orient, type, pra, spec, well, annot1, annot2) {
	pendingLines = "";
    ensureRoiManager();
    roiManager("Reset");
    run("Clear Results");

    // open original and duplicate to work copy
    open(dir + name);
    origTitle = getTitle();
    run("Duplicate...", "duplicate title=" + origTitle + "-work");
    workTitle = origTitle + "-work";

    // close original so we never touch the disk file
    selectWindow(origTitle);
    close();

    selectWindow(workTitle);
    getDimensions(width, height, channels, slices, frames);
    base = stripExt(name);

    // scale
    run("Set Scale...", "distance=0 known=0 pixel=1 unit=pixel");

    // split channels
    run("Split Channels");

    redTitle   = findChannelWindow(workTitle, "red");
    greenTitle = findChannelWindow(workTitle, "green");
    blueTitle  = findChannelWindow(workTitle, "blue");

    if (redTitle == "") {
        redTitle = findChannelWindow(workTitle, "c1");
    }
    if (greenTitle == "") {
        greenTitle = findChannelWindow(workTitle, "c2");
    }
    if (blueTitle == "") {
        blueTitle = findChannelWindow(workTitle, "c3");
    }
    
    hasRG = (redTitle != "" && greenTitle != "");
    
    // safety copies for later
    if (redTitle != "") {
        selectWindow(redTitle);
        run("Duplicate...", "title=" + base + "-red-read");
        readRedTitle = getTitle();
    }
    if (greenTitle != "") {
        selectWindow(greenTitle);
        run("Duplicate...", "title=" + base + "-green-read");
        readGreenTitle = getTitle();
    }
    if (blueTitle != "") {
        selectWindow(blueTitle);
        run("Duplicate...", "title=" + base + "-blue-read");
        readBlueTitle = getTitle();
    }

    // -------- SEGMENTATION --------
    n = 0;
    filteredMaskTitle = "";
    segTitle = "";

    if (hasRG) {
        run("Image Calculator...", "operation=Add image1=["+redTitle+"] image2=["+greenTitle+"] create");
        segTitle = getTitle();

        selectWindow(segTitle);
        run("8-bit");
        run("Subtract Background...", "rolling=20 sliding paraboloid");
        run("Subtract...", "value=40");
        run("Auto Local Threshold", "method=Phansalkar radius=2 parameter_1=0 parameter_2=0 white");
        run("Watershed");

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
    }

    // -------- STATS FOR SEGMENTED ROIs --------
    if (n > 0 && hasRG) {

		// Means on RED
        redMeans = newArray(n);
        run("Clear Results");
        selectWindow(readRedTitle);   // <--- use read-only copy
        for (r = 0; r < n; r++) {
            roiManager("Select", r);
            run("Measure");
        }
        for (r = 0; r < n; r++) {
            redMeans[r] = getResult("Mean", r);
        }

        // Means on GREEN
        greenMeans = newArray(n);
        run("Clear Results");
        selectWindow(readGreenTitle); // <--- use read-only copy
        for (r = 0; r < n; r++) {
            roiManager("Select", r);
            run("Measure");
        }
        for (r = 0; r < n; r++) {
            greenMeans[r] = getResult("Mean", r);
        }

        // Means on BLUE if present, else -1
        blueMeans = newArray(n);
        if (readBlueTitle != "") {
            run("Clear Results");
            selectWindow(readBlueTitle);   // <--- use read-only copy
            for (r = 0; r < n; r++) {
                roiManager("Select", r);
                run("Measure");
            }
            for (r = 0; r < n; r++) {
                blueMeans[r] = getResult("Mean", r);
            }
        } else {
            for (r = 0; r < n; r++) {
                blueMeans[r] = -1;
            }
        }

        // save segmented ROIs
        for (r = 0; r < n; r++) {
            segCount++;
            roiId   = "S_" + pad5(segCount);
            roiType = "segmented";
            mR = redMeans[r];
            mG = greenMeans[r];
            mB = blueMeans[r];

            appendRoiRow(folderKey, orient, type, pra, spec, well,
                         annot1, annot2, name,
                         roiId, roiType, mR, mG, mB);
        }
    }

    // -------- BUILD USER VIEW --------
    userViewTitle = workTitle; // fallback

    if (hasRG) {
        // Merge channels to one multi-channel image
        if (blueTitle != "") {
            run("Merge Channels...", "c1=["+redTitle+"] c2=["+greenTitle+"] c3=["+blueTitle+"] create");
        } else {
            run("Merge Channels...", "c1=["+redTitle+"] c2=["+greenTitle+"] create");
        }
        userViewTitle = getTitle();

        // If we have a mask, invert and multiply whole image
        if (n > 0 && filteredMaskTitle != "") {
            selectWindow(filteredMaskTitle);
            run("Invert"); // segmented objects -> 0, background -> 255
			run("Divide...", "value=255");
            run("Image Calculator...",
                "operation=Multiply image1=["+userViewTitle+"] image2=["+filteredMaskTitle+"] create");
            newView = getTitle();
            selectWindow(userViewTitle);
            close();
            userViewTitle = newView;
        }
    }

    // -------- USER-SELECTED GREEN DOTS --------
    selectWindow(userViewTitle);
    setBatchMode("show");
    selectWindow(userViewTitle);
    setTool("multipoint");
    waitForUser("Select all GREEN dots with the Multi-point tool,\nthen click OK.");
    setBatchMode("hide");

    greenPointsX = newArray(0);
    greenPointsY = newArray(0);
    nG = 0;
    if (selectionType() != -1) {
        getSelectionCoordinates(greenPointsX, greenPointsY);
        nG = greenPointsX.length;
    }

    for (i = 0; i < nG; i++) {
        x = greenPointsX[i];
        y = greenPointsY[i];
        
        valR = sampleAt(readRedTitle,   x, y);
        valG = sampleAt(readGreenTitle, x, y);
        valB = sampleAt(readBlueTitle,  x, y);

        greenCount++;
        roiId   = "G_" + pad5(greenCount);
        roiType = "user_green";

        appendRoiRow(folderKey, orient, type, pra, spec, well,
                     annot1, annot2, name,
                     roiId, roiType, valR, valG, valB);
    }

    run("Select None");

    // -------- USER-SELECTED RED DOTS --------
    setBatchMode("show");
    selectWindow(userViewTitle);
    setTool("multipoint");
    waitForUser("Select all RED dots with the Multi-point tool,\nthen click OK.");
    setBatchMode("hide");
    
    redPointsX = newArray(0);
    redPointsY = newArray(0);
    nR = 0;
    if (selectionType() != -1) {
        getSelectionCoordinates(redPointsX, redPointsY);
        nR = redPointsX.length;
    }

    for (i = 0; i < nR; i++) {
        x = redPointsX[i];
        y = redPointsY[i];

        valR = sampleAt(readRedTitle,   x, y);
        valG = sampleAt(readGreenTitle, x, y);
        valB = sampleAt(readBlueTitle,  x, y);

        redCount++;
        roiId   = "R_" + pad5(redCount);
        roiType = "user_red";

        appendRoiRow(folderKey, orient, type, pra, spec, well,
                     annot1, annot2, name,
                     roiId, roiType, valR, valG, valB);
    }    
    
    if (pendingLines != "") {
        File.append(pendingLines, outCsvPath);
    }
    // -------- CLEANUP --------
	safeClose(segTitle);
	safeClose(filteredMaskTitle);
	safeClose(redTitle);
	safeClose(greenTitle);
	safeClose(blueTitle);
	safeClose(workTitle);
	safeClose(userViewTitle);
	safeClose(readRedTitle);
	safeClose(readGreenTitle);
	safeClose(readBlueTitle);
    ensureRoiManager();
    roiManager("Reset");
    run("Clear Results");
    run("Collect Garbage");
    call("java.lang.System.gc");
}

// ----------------- CSV APPEND -----------------

function appendRoiRow(folder, orient, type, pra, spec, well,
                      annot1, annot2, imageName,
                      roiId, roiType, mR, mG, mB) {

    // function args are due to an earlier macro version
    // that kept all information. This blew up file size
    // so we only keep the relevant ones.
    line = folder + "," + imageName + "," +
           roiId + "," + roiType + "," +
           d3(mR) + "," + d3(mG) + "," + d3(mB) + "\n";

    // only store in memory for now
    pendingLines = pendingLines + line;
}


// ----------------- UTILS -----------------

function ensureRoiManager() {
    if (!isOpen("ROI Manager")) {
        run("ROI Manager...");
    }
}

function findChannelWindow(base, tag) {
    possibilities = newArray(
        base + " (" + tag + ")",
        base + "-" + tag,
        base + " " + tag,
        tag
    );
    for (ii = 0; ii < possibilities.length; ii++) {
        t = possibilities[ii];
        if (isOpen(t)) {
            return t;
        }
    }
    titles = getList("image.titles");
    for (ii = 0; ii < titles.length; ii++) {
        t = titles[ii];
        if (indexOf(toLowerCase(t), toLowerCase(base)) >= 0 &&
            indexOf(toLowerCase(t), toLowerCase(tag))  >= 0) {
            return t;
        }
    }
    return "";
}

function findNewWindow(before, after) {
    for (i = 0; i < after.length; i++) {
        exists = false;
        for (j = 0; j < before.length; j++) {
            if (after[i] == before[j]) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            return after[i];
        }
    }
    return "";
}

function stripExt(f) {
    dot = lastIndexOf(f, ".");
    if (dot < 0) {
        return f;
    }
    return substring(f, 0, dot);
}

// format helpers
function d2(x) { return toString(round(x*100)/100); }
function d3(x) { return toString(round(x*1000)/1000); }

// pad integer to 4 digits
function pad5(n) {
    if (n < 10) return "0000" + n;
    if (n < 100) return "000"  + n;
    if (n < 1000) return "00"   + n;
    if (n < 10000) return "0"   + n;
    return "" + n;
}

// sample pixel from given window, returns -1 if title is empty
function sampleAt(title, x, y) {
    if (title == "") {
        return -1;
    }

    // remember which image was active
    current = "";
    if (nImages > 0) {
        current = getTitle();
    }

    // go to the sampling image
    selectWindow(title);
    v = getPixel(x, y);

    // go back to the previous image if still open
    if (current != "" && isOpen(current)) {
        selectWindow(current);
    }

    return v;
}

function safeClose(title) {
    if (title != "" && isOpen(title)) {
        selectWindow(title);
        close();
    }
}