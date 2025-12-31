// ======================
// ask user for image folder
// ======================
imgDir = getDirectory("Select folder with JPEG tiles");
if (imgDir == "") exit("No folder selected.");

// parent folder for CSV
sep = File.separator;
idx = lastIndexOf(imgDir, sep);
if (idx == -1)
    parentDir = imgDir;         // fallback
else
    parentDir = substring(imgDir, 0, idx);

csvPath = parentDir + sep + "segmentation_ratings.csv";

// ======================
// Helper: read already scored tiles from CSV
// ======================
function loadDoneSet(csvPath) {
    done = newArray(0);
    if (File.exists(csvPath)) {
        content = File.openAsString(csvPath);
        lines = split(content, "\n");
        // header: image_no,tile_no,best_method
        for (i = 1; i < lines.length; i++) {
            line = trim(lines[i]);
            if (line == "") continue;
            parts = split(line, ",");
            if (parts.length >= 2) {
                key = parts[0] + "_" + parts[1];
                done = Array.concat(done, key);
            }
        }
    } else {
        // create file with header
        File.append("image_no,tile_no,best_method\n", csvPath);
    }
    return done;
}

function isDone(key, doneArr) {
    for (i = 0; i < doneArr.length; i++) {
        if (doneArr[i] == key)
            return true;
    }
    return false;
}

// ALT = left, SPACE = right, SHIFT = quit
function waitForChoiceAltSpace() {
    print("ALT = LEFT segmentation, SPACE = RIGHT segmentation, SHIFT = quit.");

    wasAlt   = false;
    wasSpace = false;
    wasShift = false;

    while (true) {
        altDown   = isKeyDown("alt");
        spaceDown = isKeyDown("space");
        shiftDown = isKeyDown("shift");

        if (altDown && !wasAlt)
            return 1;   // left
        if (spaceDown && !wasSpace)
            return 2;   // right
        if (shiftDown && !wasShift)
            return -1;  // quit

        wasAlt   = altDown;
        wasSpace = spaceDown;
        wasShift = shiftDown;

        wait(20);
    }
}

function trim(s) {
    s = replace(s, "^[ \t\r\n]+", "");
    s = replace(s, "[ \t\r\n]+$", "");
    return s;
}

// === MAIN ===
setBatchMode(false);

if (!File.isDirectory(imgDir)) {
    exit("imgDir is not a directory: " + imgDir);
}

done = loadDoneSet(csvPath);
files = getFileList(imgDir);

for (i = 0; i < files.length; i++) {
    name = files[i];

    if (!endsWith(name, ".jpg")) continue;
    if (indexOf(name, "_imageJ") != -1) continue;
    if (indexOf(name, "_unet") != -1) continue;

    base = substring(name, 0, lengthOf(name) - 4); // remove .jpg

    us = indexOf(base, "_");
    if (us < 0) {
        print("Skipping file without underscore: " + name);
        continue;
    }
    image_no = substring(base, 0, us);
    tile_no  = substring(base, us + 1);

    key = image_no + "_" + tile_no;
    if (isDone(key, done)) {
        print("Skipping already scored: " + key);
        continue;
    }

    imgPath   = imgDir + name;
    imgjPath  = imgDir + base + "_imageJ.jpg";
    unetPath  = imgDir + base + "_unet.jpg";

    if (!File.exists(imgjPath)) {
        print("Missing ImageJ mask for " + base + ", expected " + imgjPath);
        continue;
    }
    if (!File.exists(unetPath)) {
        print("Missing U-Net mask for " + base + ", expected " + unetPath);
        continue;
    }

    // open original + masks
    open(imgPath);
    origTitle = getTitle();
    rename("IMAGE");

    open(imgjPath);
    imgjTitle = getTitle();

    open(unetPath);
    unetTitle = getTitle();

    // randomize which mask is left/right
    r = floor(getRandom() * 2);
    if (r == 0) {
        leftTitle   = imgjTitle;
        leftMethod  = "imageJ";
        rightTitle  = unetTitle;
        rightMethod = "unet";
    } else {
        leftTitle   = unetTitle;
        leftMethod  = "unet";
        rightTitle  = imgjTitle;
        rightMethod = "imageJ";
    }

    selectWindow(leftTitle);  rename("SEG_LEFT");
    selectWindow(rightTitle); rename("SEG_RIGHT");

    // arrange windows
    selectWindow("SEG_LEFT");
    getLocation(x, y);
    getDimensions(w, h, c, z, t);
    setLocation(0, 0);

    selectWindow("SEG_RIGHT");
    setLocation(w + 10, 0);

    selectWindow("IMAGE");
    setLocation(0, h + 40);

    print("Scoring image " + image_no + ", tile " + tile_no +
          "  (LEFT=" + leftMethod + ", RIGHT=" + rightMethod + ")");

    choice = waitForChoiceAltSpace();
    if (choice == -1) {
        print("Stopping on user request (SHIFT).");
        exit("Stopped by user.");
    }

    if (choice == 1)
        bestMethod = leftMethod;   // ALT
    else
        bestMethod = rightMethod;  // SPACE

    line = image_no + "," + tile_no + "," + bestMethod + "\n";
    File.append(line, csvPath);
    done = Array.concat(done, key);

    selectWindow("IMAGE");     close();
    selectWindow("SEG_LEFT");  close();
    selectWindow("SEG_RIGHT"); close();
}

print("All tiles processed (or none left to score).");
