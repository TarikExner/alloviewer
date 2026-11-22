/*
 * AlloViewer Circular Ground-Truth Annotator
 * Version 2.5.0 (2026-10-30)
 *
 * Reliability-first Fiji/ImageJ macro for manual cell ground truth.
 * There is no installer, no toolbar modification and no LUT manipulation.
 * Drag this file into Fiji and click Run once in the Script Editor.
 *
 * CONTROLS WHILE THE IMAGE WINDOW IS ACTIVE
 * -----------------------------------------
 *   Click or drag-release          add GREEN circular ROI
 *   Shift + click/drag-release     add ORANGE circular ROI
 *   Shift + Option + click/drag    add UNCERTAIN circular ROI
 *   Option/Alt + click             undo the most recent ROI
 *   Hold Space + left-drag         pan the zoomed image
 *   Hold Shift+Option+Space (0.6s) complete image and open next
 *   Hold Option + Space (0.6 s)    save draft and pause
 *   Space tap / Shift + Space      no action (safety lock)
 *
 * A fixed-diameter circle follows the cursor while the mouse button is held.
 * Every edit is immediately saved as a draft. One completed CSV is written
 * per image. Circular ROIs may cross an image boundary; only their visible
 * pixels are used for intensity statistics. Images open maximized at 600%
 * zoom by default. A 3 x 3 coverage reminder helps prevent skipped regions.
 * One image-summary row is also saved for every completed image, including
 * images with zero annotated cells.
 *
 * INTENSITY STATISTICS
 * --------------------
 * For ordinary 24-bit RGB images, the packed red, green and blue values are
 * measured directly. For 8/16/32-bit multichannel images, raw channels 1, 2
 * and 3 are measured independently and written to the red, green and blue
 * columns respectively. Display ranges and LUTs do not change these values.
 */

var AV_VERSION = "2.5.0";
var AV_MAX_IMAGES = 100000;
var AV_MAX_CELLS = 10000;
var AV_COMMAND_HOLD_MS = 600;

var AV_IMAGE_ROOT = "";
var AV_OUTPUT_ROOT = "";
var AV_CSV_DIR = "";
var AV_DRAFT_DIR = "";
var AV_SUMMARY_DIR = "";
var AV_ANNOTATOR = "annotator";
var AV_DIAMETER = 7;
var AV_TARGET_ZOOM_PERCENT = 600;
var AV_ACTUAL_ZOOM_PERCENT = 600;

var AV_IMAGE_PATHS = newArray(0);
var AV_IMAGE_RELS = newArray(0);
var AV_IMAGE_KEYS = newArray(0);
var AV_IMAGE_COUNT = 0;

var AV_CURRENT_INDEX = -1;
var AV_IMAGE_PATH = "";
var AV_IMAGE_REL = "";
var AV_IMAGE_NAME = "";
var AV_RECORD_KEY = "";
var AV_IMAGE_ID = 0;
var AV_IMAGE_WIDTH = 0;
var AV_IMAGE_HEIGHT = 0;
var AV_IMAGE_CHANNELS = 0;
var AV_IMAGE_BIT_DEPTH = 0;
var AV_FINAL_PATH = "";
var AV_DRAFT_PATH = "";
var AV_SUMMARY_PATH = "";

var AV_X = newArray(0);
var AV_Y = newArray(0);
var AV_SIZE = newArray(0);
var AV_CLASS = newArray(0);
var AV_COUNT = 0;
var AV_COVERAGE = newArray(0);
var AV_COVERAGE_COUNT = 0;
var AV_FINISH_ARMED = 0;

var AV_STAT_N = 0;
var AV_STAT_R_MEAN = NaN;
var AV_STAT_R_SD = NaN;
var AV_STAT_R_MIN = NaN;
var AV_STAT_R_MAX = NaN;
var AV_STAT_R_SUM = NaN;
var AV_STAT_G_MEAN = NaN;
var AV_STAT_G_SD = NaN;
var AV_STAT_G_MIN = NaN;
var AV_STAT_G_MAX = NaN;
var AV_STAT_G_SUM = NaN;
var AV_STAT_B_MEAN = NaN;
var AV_STAT_B_SD = NaN;
var AV_STAT_B_MIN = NaN;
var AV_STAT_B_MAX = NaN;
var AV_STAT_B_SUM = NaN;

// Scratch values populated by computeActiveChannelStatistics(). They are
// global because ImageJ macro functions cannot return several numeric values.
var AV_CHANNEL_STAT_N = 0;
var AV_CHANNEL_STAT_MEAN = NaN;
var AV_CHANNEL_STAT_SD = NaN;
var AV_CHANNEL_STAT_MIN = NaN;
var AV_CHANNEL_STAT_MAX = NaN;
var AV_CHANNEL_STAT_SUM = NaN;


macro "Run AlloViewer Circular Ground-Truth Annotator v2.5.0" {
    runAnnotator();
}


function runAnnotator() {
    requires("1.53v");

    AV_IMAGE_ROOT = getDirectory("Choose the folder containing the images");
    if (AV_IMAGE_ROOT=="")
        exit("No image folder was selected.");

    AV_IMAGE_PATHS = newArray(AV_MAX_IMAGES);
    AV_IMAGE_RELS = newArray(AV_MAX_IMAGES);
    AV_IMAGE_KEYS = newArray(AV_MAX_IMAGES);
    AV_IMAGE_COUNT = 0;
    scanImageTree(AV_IMAGE_ROOT, "");

    if (AV_IMAGE_COUNT==0) {
        showMessage("AlloViewer: no images found",
            "No supported image files were found in:\n" + AV_IMAGE_ROOT +
            "\n\nSubfolders were searched as well.\n"+
            "Supported extensions: PNG, TIF, TIFF, JPG, JPEG, BMP and GIF.");
        return;
    }

    Dialog.create("AlloViewer circular ground truth");
    Dialog.addString("Annotator name or ID:", "annotator", 24);
    Dialog.addNumber("Circular ROI diameter (pixels):", 7);
    Dialog.addNumber("Annotation zoom (percent):", 600);
    Dialog.addMessage(
        "Use a separate source folder for each annotator if several people work simultaneously.\n"+
        "The annotator ID is recorded in every ROI row.\n"+
        "Recommended zoom: 600%. Use 800% only if you prefer more panning."
    );
    Dialog.show();
    AV_ANNOTATOR = Dialog.getString();
    AV_DIAMETER = round(Dialog.getNumber());
    AV_TARGET_ZOOM_PERCENT = round(Dialog.getNumber());
    if (AV_ANNOTATOR=="") AV_ANNOTATOR = "annotator";
    if (AV_DIAMETER<1) AV_DIAMETER = 7;
    if (AV_TARGET_ZOOM_PERCENT<100) AV_TARGET_ZOOM_PERCENT = 600;
    if (AV_TARGET_ZOOM_PERCENT>3200) AV_TARGET_ZOOM_PERCENT = 3200;

    AV_OUTPUT_ROOT = AV_IMAGE_ROOT + "AlloViewer_ground_truth" + File.separator;
    AV_CSV_DIR = AV_OUTPUT_ROOT + "per_image_csv" + File.separator;
    AV_DRAFT_DIR = AV_OUTPUT_ROOT + "drafts" + File.separator;
    AV_SUMMARY_DIR = AV_OUTPUT_ROOT + "per_image_summary" + File.separator;
    makeDirectoryIfNeeded(AV_OUTPUT_ROOT);
    makeDirectoryIfNeeded(AV_CSV_DIR);
    makeDirectoryIfNeeded(AV_DRAFT_DIR);
    makeDirectoryIfNeeded(AV_SUMMARY_DIR);

    // v2.1.x used a header-only cell CSV for an empty image. Those files are
    // still valid completion markers. Create a separate image-level summary
    // for any older completed CSV so empty images survive later combination.
    ensureCompletedImageSummaries();

    completed = countCompletedImages();
    remaining = AV_IMAGE_COUNT-completed;

    showMessage("AlloViewer circular annotator",
        "Images found: " + AV_IMAGE_COUNT +
        "\nAlready complete: " + completed +
        "\nRemaining: " + remaining +
        "\n\nClick or drag-release: GREEN"+
        "\nShift + click/drag-release: ORANGE"+
        "\nShift + Option + click/drag-release: UNCERTAIN"+
        "\nOption/Alt + click: undo"+
        "\nHold Space + left-drag: pan at high zoom"+
        "\nHold Shift + Option + Space for 0.6 s: complete image and open next"+
        "\nHold Option + Space for 0.6 s: save draft and pause"+
        "\nSpace tap or Shift + Space: no action (safety lock)"+
        "\n\nCircular ROI diameter: " + AV_DIAMETER + " px"+
        "\nAnnotation zoom target: " + AV_TARGET_ZOOM_PERCENT + "%"+
        "\nA first held Shift + Option + Space warns if a 3 x 3 region was not visited."+
        "\nHold Shift + Option + Space again to finish anyway."+
        "\nThe source image pixels are never changed."+
        "\nRaw channels 1/2/3 are measured separately in multichannel TIFFs.");

    while (1) {
        nextIndex = findNextImageIndex();
        if (nextIndex<0) {
            buildCombinedResults();
            showMessage("AlloViewer: annotation complete",
                "All " + AV_IMAGE_COUNT + " discovered images are complete.\n\n"+
                "Per-image CSV files:\n" + AV_CSV_DIR +
                "\n\nCombined cell CSV:\n" + AV_OUTPUT_ROOT + "combined_results.csv"+
                "\n\nCombined image summary:\n" + AV_OUTPUT_ROOT + "combined_image_summary.csv");
            return;
        }

        if (!openImageAtIndex(nextIndex)) {
            showMessage("AlloViewer: image opening failed",
                "Fiji could not open:\n" + AV_IMAGE_PATH +
                "\n\nThe session was paused. No image was marked complete.");
            return;
        }

        resetAnnotationArrays();
        loadDraft();
        drawAllAnnotations();
        saveDraft();
        applyComfortableZoom();
        setTool("point");

        action = annotateCurrentImage();
        if (action=="finish") {
            if (!saveCompletedAnnotation()) {
                saveDraft();
                closeCurrentImage();
                showMessage("AlloViewer save error",
                    "The per-image CSV could not be created.\n"+
                    "The draft was retained and the image was not marked complete.");
                return;
            }
            closeCurrentImage();
        } else {
            saveDraft();
            closeCurrentImage();
            showMessage("AlloViewer paused",
                "The current image was saved as a draft.\n"+
                "Run the macro again to resume it.");
            return;
        }
    }
}


function scanImageTree(directory, relativePrefix) {
    entries = getFileList(directory);
    Array.sort(entries);
    for (scanI=0; scanI<entries.length; scanI++) {
        entry = entries[scanI];
        if (endsWith(entry, "/")) {
            if (entry=="AlloViewer_ground_truth/" || entry=="AlloViewer_annotations/")
                continue;
            scanImageTree(directory+entry, relativePrefix+entry);
        } else if (isSupportedImage(entry)) {
            if (AV_IMAGE_COUNT>=AV_MAX_IMAGES)
                exit("More than " + AV_MAX_IMAGES + " images were found. Increase AV_MAX_IMAGES.");
            relativePath = relativePrefix+entry;
            AV_IMAGE_PATHS[AV_IMAGE_COUNT] = directory+entry;
            AV_IMAGE_RELS[AV_IMAGE_COUNT] = relativePath;
            AV_IMAGE_KEYS[AV_IMAGE_COUNT] = makeRecordKey(relativePath);
            AV_IMAGE_COUNT++;
        }
    }
}


function isSupportedImage(name) {
    lowerName = toLowerCase(name);
    // Explicit checks are intentional. A chained logical expression after a
    // return statement is evaluated incorrectly by some ImageJ 1 macro builds.
    if (endsWith(lowerName, ".tif")) return 1;
    if (endsWith(lowerName, ".tiff")) return 1;
    if (endsWith(lowerName, ".png")) return 1;
    if (endsWith(lowerName, ".jpg")) return 1;
    if (endsWith(lowerName, ".jpeg")) return 1;
    if (endsWith(lowerName, ".bmp")) return 1;
    if (endsWith(lowerName, ".gif")) return 1;
    return 0;
}


function makeRecordKey(relativePath) {
    imageFileName = File.getName(relativePath);
    baseName = File.getNameWithoutExtension(imageFileName);
    baseName = replace(baseName, " ", "_");
    baseName = replace(baseName, ":", "_");
    baseName = replace(baseName, ",", "_");
    checksum = IJ.checksum("MD5 string", relativePath);
    if (lengthOf(checksum)>12) checksum = substring(checksum, 0, 12);
    return baseName + "__" + checksum;
}


function countCompletedImages() {
    completedCount = 0;
    for (countI=0; countI<AV_IMAGE_COUNT; countI++) {
        recordPath = AV_CSV_DIR + AV_IMAGE_KEYS[countI] + ".csv";
        if (File.exists(recordPath)) completedCount++;
    }
    return completedCount;
}


function findNextImageIndex() {
    for (findI=0; findI<AV_IMAGE_COUNT; findI++) {
        recordPath = AV_CSV_DIR + AV_IMAGE_KEYS[findI] + ".csv";
        if (!File.exists(recordPath)) return findI;
    }
    return -1;
}


function openImageAtIndex(imageIndex) {
    AV_CURRENT_INDEX = imageIndex;
    AV_IMAGE_PATH = AV_IMAGE_PATHS[imageIndex];
    AV_IMAGE_REL = AV_IMAGE_RELS[imageIndex];
    AV_IMAGE_NAME = File.getName(AV_IMAGE_PATH);
    AV_RECORD_KEY = AV_IMAGE_KEYS[imageIndex];
    AV_FINAL_PATH = AV_CSV_DIR + AV_RECORD_KEY + ".csv";
    AV_DRAFT_PATH = AV_DRAFT_DIR + AV_RECORD_KEY + ".wip.tsv";
    AV_SUMMARY_PATH = AV_SUMMARY_DIR + AV_RECORD_KEY + ".csv";

    imageCountBefore = nImages;
    IJ.redirectErrorMessages();
    open(AV_IMAGE_PATH);
    if (nImages<=imageCountBefore) return 0;

    AV_IMAGE_ID = getImageID();
    getDimensions(AV_IMAGE_WIDTH, AV_IMAGE_HEIGHT, AV_IMAGE_CHANNELS, slices, frames);
    AV_IMAGE_BIT_DEPTH = bitDepth();
    return 1;
}


function annotateCurrentImage() {
    waitUntilMouseAndSpaceReleased();
    wasMouseDown = 0;
    pressAction = "green";
    lastX = 0;
    lastY = 0;
    statusTick = 0;
    AV_FINISH_ARMED = 0;
    updateStatusLine();

    while (1) {
        if (!isOpen(AV_IMAGE_ID)) return "pause";

        if (!isActive(AV_IMAGE_ID)) {
            showStatus("!Click the annotation image to continue: " + AV_IMAGE_NAME);
            wait(20);
            continue;
        }

        getCursorLoc(mx, my, mz, flags);
        mouseDown = ((flags&16)!=0);
        spaceDown = isKeyDown("space");
        if (updateCoverage(mx, my)) AV_FINISH_ARMED = 0;

        // A Space gesture is handled as one unit. Plain Space never advances:
        // it only supports ImageJ's native Space-drag panning. Finishing and
        // pausing require a modifier plus a deliberate hold. Consuming the
        // whole gesture also prevents pan-release becoming an annotation.
        if (spaceDown) {
            shiftedSpace = isKeyDown("shift");
            optionSpace = isKeyDown("alt");
            spaceStartedAt = getTime();
            spaceWasDragged = mouseDown;

            while (isKeyDown("space")) {
                if (!isOpen(AV_IMAGE_ID)) return "pause";
                getCursorLoc(panX, panY, panZ, panFlags);
                if ((panFlags&16)!=0) spaceWasDragged = 1;
                if (updateCoverage(panX, panY)) AV_FINISH_ARMED = 0;
                wait(10);
            }

            // Also consume the end of a drag if Space happened to be released
            // before the mouse button.
            getCursorLoc(panX, panY, panZ, panFlags);
            while ((panFlags&16)!=0) {
                spaceWasDragged = 1;
                if (updateCoverage(panX, panY)) AV_FINISH_ARMED = 0;
                wait(10);
                if (!isOpen(AV_IMAGE_ID)) return "pause";
                getCursorLoc(panX, panY, panZ, panFlags);
            }

            spaceHeldMs = getTime()-spaceStartedAt;

            wasMouseDown = 0;
            if (spaceWasDragged) {
                AV_FINISH_ARMED = 0;
                updateStatusLine();
                continue;
            }

            if (shiftedSpace && optionSpace) {
                if (spaceHeldMs<AV_COMMAND_HOLD_MS) {
                    beep();
                    showStatus("!Finish not triggered: hold Shift + Option + Space for 0.6 seconds.");
                    wait(250);
                    continue;
                }

                if (AV_COVERAGE_COUNT<9 && !AV_FINISH_ARMED) {
                    beep();
                    AV_FINISH_ARMED = 1;
                    showStatus("!Coverage " + AV_COVERAGE_COUNT + "/9; still unvisited: " +
                        missingCoverageText() + ". Hold Shift + Option + Space again for 0.6 s to finish anyway.");
                    wait(250);
                    continue;
                }
                return "finish";
            }

            if (optionSpace) {
                if (spaceHeldMs>=AV_COMMAND_HOLD_MS)
                    return "pause";
                beep();
                showStatus("!Pause not triggered: hold Option + Space for 0.6 seconds.");
                wait(250);
                continue;
            }

            if (shiftedSpace) {
                showStatus("!Shift + Space is safety-locked. Add Option and hold all three keys for 0.6 seconds to finish.");
                wait(150);
                continue;
            }

            showStatus("!Space tap ignored for safety. Hold Shift + Option + Space for 0.6 seconds to finish.");
            wait(150);
            continue;
        }

        if (mouseDown && !wasMouseDown) {
            lastX = mx;
            lastY = my;
            if ((flags&8)!=0) {
                if ((flags&1)!=0)
                    pressAction = "uncertain";
                else
                    pressAction = "undo";
            } else if ((flags&1)!=0)
                pressAction = "orange";
            else
                pressAction = "green";
        }

        if (mouseDown && pressAction!="undo") {
            lastX = mx;
            lastY = my;
            showPreview(lastX, lastY, pressAction);
        }

        if (!mouseDown && wasMouseDown) {
            if (pressAction=="undo")
                undoLastCell();
            else
                addCell(lastX, lastY, pressAction);
            AV_FINISH_ARMED = 0;
            updateStatusLine();
        }

        wasMouseDown = mouseDown;
        statusTick++;
        if ((statusTick%50)==0) updateStatusLine();
        wait(10);
    }
}


function showPreview(centerX, centerY, cellClass) {
    left = circleLeft(centerX, AV_DIAMETER);
    top = circleTop(centerY, AV_DIAMETER);
    makeOval(left, top, AV_DIAMETER, AV_DIAMETER);
    if (cellClass=="green")
        Roi.setStrokeColor("00ff00");
    else if (cellClass=="orange")
        Roi.setStrokeColor("ff8800");
    else
        Roi.setStrokeColor("00ccff");
    Roi.setStrokeWidth(1.5);
}


function addCell(centerX, centerY, cellClass) {
    if (AV_COUNT>=AV_MAX_CELLS) {
        showMessage("AlloViewer", "The safety limit of " + AV_MAX_CELLS + " cells was reached.");
        return;
    }

    if (!centerCloseToImage(centerX, centerY, AV_DIAMETER)) {
        beep();
        showStatus("!The circle centre is too far outside the image; no ROI was added.");
        return;
    }

    left = circleLeft(centerX, AV_DIAMETER);
    top = circleTop(centerY, AV_DIAMETER);
    AV_X[AV_COUNT] = left;
    AV_Y[AV_COUNT] = top;
    AV_SIZE[AV_COUNT] = AV_DIAMETER;
    AV_CLASS[AV_COUNT] = cellClass;
    AV_COUNT++;

    makeOval(left, top, AV_DIAMETER, AV_DIAMETER);
    if (cellClass=="green")
        Overlay.addSelection("00ff00", 1.5);
    else if (cellClass=="orange")
        Overlay.addSelection("ff8800", 1.5);
    else
        Overlay.addSelection("00ccff", 1.5);
    Overlay.show;
    Overlay.selectable(false);
    run("Select None");
    saveDraft();
}


function undoLastCell() {
    if (AV_COUNT<=0) {
        beep();
        return;
    }
    AV_COUNT--;
    if (Overlay.size>AV_COUNT)
        Overlay.removeSelection(AV_COUNT);
    Overlay.show;
    run("Select None");
    saveDraft();
}


function drawAllAnnotations() {
    Overlay.clear;
    for (drawI=0; drawI<AV_COUNT; drawI++) {
        makeOval(AV_X[drawI], AV_Y[drawI], AV_SIZE[drawI], AV_SIZE[drawI]);
        if (AV_CLASS[drawI]=="green")
            Overlay.addSelection("00ff00", 1.5);
        else if (AV_CLASS[drawI]=="orange")
            Overlay.addSelection("ff8800", 1.5);
        else
            Overlay.addSelection("00ccff", 1.5);
    }
    Overlay.drawLabels(false);
    Overlay.selectable(false);
    Overlay.show;
    run("Select None");
}


function updateStatusLine() {
    greenCount = countClass("green");
    orangeCount = countClass("orange");
    uncertainCount = countClass("uncertain");
    if (AV_FINISH_ARMED) {
        showStatus("!Coverage " + AV_COVERAGE_COUNT + "/9; still unvisited: " +
            missingCoverageText() + ". Hold Shift + Option + Space again for 0.6 s to finish anyway.");
        return;
    }
    showStatus("!" + AV_IMAGE_REL + " | green=" + greenCount +
        " orange=" + orangeCount + " uncertain=" + uncertainCount +
        " | circle=" + AV_DIAMETER +
        "px | coverage=" + AV_COVERAGE_COUNT + "/9 | click=green, Shift-click=orange, " +
        "Shift-Option-click=uncertain, Option-click=undo, Space-drag=pan, hold Shift-Option-Space=next");
}


function countClass(cellClass) {
    classCount = 0;
    for (classI=0; classI<AV_COUNT; classI++)
        if (AV_CLASS[classI]==cellClass) classCount++;
    return classCount;
}


function circleLeft(centerX, diameter) {
    return round(centerX)-floor(diameter/2);
}


function circleTop(centerY, diameter) {
    return round(centerY)-floor(diameter/2);
}


function centerCloseToImage(centerX, centerY, diameter) {
    // ImageJ 1.54p's macro language has floor() but no ceil(). Diameter is
    // always an integer here, so this is the compatible ceiling calculation.
    margin = floor((diameter+1)/2);
    if (centerX < -margin) return 0;
    if (centerY < -margin) return 0;
    if (centerX >= AV_IMAGE_WIDTH+margin) return 0;
    if (centerY >= AV_IMAGE_HEIGHT+margin) return 0;
    return 1;
}


function updateCoverage(cursorX, cursorY) {
    if (cursorX<0) return 0;
    if (cursorY<0) return 0;
    if (cursorX>=AV_IMAGE_WIDTH) return 0;
    if (cursorY>=AV_IMAGE_HEIGHT) return 0;

    coverageColumn = floor(3*cursorX/AV_IMAGE_WIDTH);
    coverageRow = floor(3*cursorY/AV_IMAGE_HEIGHT);
    if (coverageColumn>2) coverageColumn = 2;
    if (coverageRow>2) coverageRow = 2;
    coverageIndex = 3*coverageRow+coverageColumn;
    if (AV_COVERAGE[coverageIndex]==0) {
        AV_COVERAGE[coverageIndex] = 1;
        AV_COVERAGE_COUNT++;
        return 1;
    }
    return 0;
}


function missingCoverageText() {
    coverageLabels = newArray("top-left", "top-centre", "top-right",
        "middle-left", "centre", "middle-right",
        "bottom-left", "bottom-centre", "bottom-right");
    missingText = "";
    for (coverageI=0; coverageI<9; coverageI++) {
        if (AV_COVERAGE[coverageI]!=0) continue;
        if (missingText!="") missingText = missingText + ", ";
        missingText = missingText + coverageLabels[coverageI];
    }
    return missingText;
}


function waitUntilMouseAndSpaceReleased() {
    if (!isOpen(AV_IMAGE_ID)) return;
    getCursorLoc(waitX, waitY, waitZ, waitFlags);
    while (((waitFlags&16)!=0) || isKeyDown("space")) {
        wait(20);
        if (!isOpen(AV_IMAGE_ID)) return;
        getCursorLoc(waitX, waitY, waitZ, waitFlags);
    }
}


function applyComfortableZoom() {
    // Begin from a predictable fit, then use ImageJ's native zoom steps until
    // the requested magnification is reached. At 600%, a 7 px circle is about
    // 42 screen pixels across, while ImageJ's Space-drag pans the canvas.
    run("Maximize");
    wait(150);
    run("Scale to Fit");
    wait(100);
    zoomSteps = 0;
    targetMagnification = AV_TARGET_ZOOM_PERCENT/100;
    while (getZoom()<targetMagnification-0.01 && zoomSteps<30) {
        run("In [+]");
        zoomSteps++;
        wait(25);
    }
    AV_ACTUAL_ZOOM_PERCENT = round(getZoom()*100);
}


function closeCurrentImage() {
    if (!isOpen(AV_IMAGE_ID)) return;
    selectImage(AV_IMAGE_ID);
    Overlay.remove;
    run("Select None");
    setOption("Changes", false);
    close();
}


function resetAnnotationArrays() {
    AV_X = newArray(AV_MAX_CELLS);
    AV_Y = newArray(AV_MAX_CELLS);
    AV_SIZE = newArray(AV_MAX_CELLS);
    AV_CLASS = newArray(AV_MAX_CELLS);
    AV_COUNT = 0;
    AV_COVERAGE = newArray(9);
    for (resetI=0; resetI<9; resetI++) AV_COVERAGE[resetI] = 0;
    AV_COVERAGE_COUNT = 0;
    AV_FINISH_ARMED = 0;
}


function saveDraft() {
    draftText = "roi_x_px\troi_y_px\tdiameter_px\tclass\n";
    for (draftI=0; draftI<AV_COUNT; draftI++) {
        // Without the leading empty string, ImageJ 1.54p treats this row as
        // numeric because its first field is the X coordinate and writes NaN.
        draftRow = "" + AV_X[draftI] + "\t" + AV_Y[draftI] + "\t" +
            AV_SIZE[draftI] + "\t" + AV_CLASS[draftI] + "\n";
        draftText = "" + draftText + draftRow;
    }
    File.saveString(draftText, AV_DRAFT_PATH);
}


function loadDraft() {
    if (!File.exists(AV_DRAFT_PATH)) return;
    draftText = replace(File.openAsString(AV_DRAFT_PATH), "\r", "");
    draftLines = split(draftText, "\n");
    for (draftI=1; draftI<draftLines.length && AV_COUNT<AV_MAX_CELLS; draftI++) {
        if (trim(draftLines[draftI])=="") continue;
        draftCols = split(draftLines[draftI], "\t");
        if (draftCols.length<4) continue;
        draftX = parseFloat(draftCols[0]);
        draftY = parseFloat(draftCols[1]);
        draftSize = parseFloat(draftCols[2]);
        draftClass = trim(draftCols[3]);
        if (isNaN(draftX) || isNaN(draftY) || isNaN(draftSize)) continue;
        if (draftClass!="green" && draftClass!="orange" && draftClass!="uncertain") continue;
        AV_X[AV_COUNT] = draftX;
        AV_Y[AV_COUNT] = draftY;
        AV_SIZE[AV_COUNT] = draftSize;
        AV_CLASS[AV_COUNT] = draftClass;
        AV_COUNT++;
    }
}


function completedCsvHeader() {
    return "image_name,image_relative_path,cell_id,annotation_class,annotator,"+
        "roi_shape,center_x_px,center_y_px,roi_x_px,roi_y_px,diameter_px,"+
        "radius_px,mask_pixel_count,image_width_px,image_height_px,"+
        "image_channels,image_bit_depth,red_mean,red_sd,red_min,red_max,"+
        "red_sum,green_mean,green_sd,green_min,green_max,green_sum,"+
        "blue_mean,blue_sd,blue_min,blue_max,blue_sum,completed_at\n";
}


function saveCompletedAnnotation() {
    completedAt = timeStamp();
    csvText = completedCsvHeader();

    for (saveI=0; saveI<AV_COUNT; saveI++) {
        computeCircleStatistics(AV_X[saveI], AV_Y[saveI], AV_SIZE[saveI]);
        centerX = AV_X[saveI] + AV_SIZE[saveI]/2;
        centerY = AV_Y[saveI] + AV_SIZE[saveI]/2;
        radius = AV_SIZE[saveI]/2;

        // The explicit empty-string prefix is required by ImageJ 1.54p. With
        // a string-returning user function as the first term of a += chain,
        // that interpreter can incorrectly demand a numeric return value.
        rowText = "" + csvQuote(AV_IMAGE_NAME) + "," + csvQuote(AV_IMAGE_REL) + "," +
            (saveI+1) + "," + csvQuote(AV_CLASS[saveI]) + "," +
            csvQuote(AV_ANNOTATOR) + "," + csvQuote("circle") + "," +
            numberText(centerX) + "," + numberText(centerY) + "," +
            AV_X[saveI] + "," + AV_Y[saveI] + "," + AV_SIZE[saveI] + "," +
            numberText(radius) + "," + AV_STAT_N + "," + AV_IMAGE_WIDTH + "," +
            AV_IMAGE_HEIGHT + "," + AV_IMAGE_CHANNELS + "," + AV_IMAGE_BIT_DEPTH + "," +
            numberText(AV_STAT_R_MEAN) + "," + numberText(AV_STAT_R_SD) + "," +
            numberText(AV_STAT_R_MIN) + "," + numberText(AV_STAT_R_MAX) + "," +
            numberText(AV_STAT_R_SUM) + "," + numberText(AV_STAT_G_MEAN) + "," +
            numberText(AV_STAT_G_SD) + "," + numberText(AV_STAT_G_MIN) + "," +
            numberText(AV_STAT_G_MAX) + "," + numberText(AV_STAT_G_SUM) + "," +
            numberText(AV_STAT_B_MEAN) + "," + numberText(AV_STAT_B_SD) + "," +
            numberText(AV_STAT_B_MIN) + "," + numberText(AV_STAT_B_MAX) + "," +
            numberText(AV_STAT_B_SUM) + "," + csvQuote(completedAt) + "\n";
        csvText = "" + csvText + rowText;
    }

    // Write the image-level row first. The per-cell file remains the completion
    // marker, so a failed summary write can never silently mark an image done.
    if (!saveImageSummary(completedAt)) return 0;
    File.saveString(csvText, AV_FINAL_PATH);
    if (!File.exists(AV_FINAL_PATH)) return 0;
    if (File.exists(AV_DRAFT_PATH)) File.delete(AV_DRAFT_PATH);
    return 1;
}


function computeCircleStatistics(left, top, diameter) {
    // A 24-bit RGB image stores all three colour values in one packed pixel.
    // A multichannel 8/16/32-bit image instead stores one raw value per
    // channel. Measuring the active channel three times is essential: copying
    // one 16-bit value into all colour columns would destroy channel-specific
    // fluorescence information.
    if (AV_IMAGE_BIT_DEPTH==24) {
        computePackedRgbStatistics(left, top, diameter);
        return;
    }

    if (AV_IMAGE_CHANNELS>=2) {
        Stack.getPosition(savedChannel, savedSlice, savedFrame);

        Stack.setChannel(1);
        computeActiveChannelStatistics(left, top, diameter);
        AV_STAT_N = AV_CHANNEL_STAT_N;
        AV_STAT_R_MEAN = AV_CHANNEL_STAT_MEAN;
        AV_STAT_R_SD = AV_CHANNEL_STAT_SD;
        AV_STAT_R_MIN = AV_CHANNEL_STAT_MIN;
        AV_STAT_R_MAX = AV_CHANNEL_STAT_MAX;
        AV_STAT_R_SUM = AV_CHANNEL_STAT_SUM;

        Stack.setChannel(2);
        computeActiveChannelStatistics(left, top, diameter);
        AV_STAT_G_MEAN = AV_CHANNEL_STAT_MEAN;
        AV_STAT_G_SD = AV_CHANNEL_STAT_SD;
        AV_STAT_G_MIN = AV_CHANNEL_STAT_MIN;
        AV_STAT_G_MAX = AV_CHANNEL_STAT_MAX;
        AV_STAT_G_SUM = AV_CHANNEL_STAT_SUM;

        if (AV_IMAGE_CHANNELS>=3) {
            Stack.setChannel(3);
            computeActiveChannelStatistics(left, top, diameter);
            AV_STAT_B_MEAN = AV_CHANNEL_STAT_MEAN;
            AV_STAT_B_SD = AV_CHANNEL_STAT_SD;
            AV_STAT_B_MIN = AV_CHANNEL_STAT_MIN;
            AV_STAT_B_MAX = AV_CHANNEL_STAT_MAX;
            AV_STAT_B_SUM = AV_CHANNEL_STAT_SUM;
        } else {
            AV_STAT_B_MEAN = NaN;
            AV_STAT_B_SD = NaN;
            AV_STAT_B_MIN = NaN;
            AV_STAT_B_MAX = NaN;
            AV_STAT_B_SUM = NaN;
        }

        // Channel selection changes only the view, not source pixels. Restore
        // the channel that was active when saving began.
        Stack.setChannel(savedChannel);
        return;
    }

    // A true one-channel image is grayscale. Repeating its one raw intensity
    // in the three legacy colour columns preserves the existing CSV schema.
    computeActiveChannelStatistics(left, top, diameter);
    AV_STAT_N = AV_CHANNEL_STAT_N;
    AV_STAT_R_MEAN = AV_CHANNEL_STAT_MEAN;
    AV_STAT_R_SD = AV_CHANNEL_STAT_SD;
    AV_STAT_R_MIN = AV_CHANNEL_STAT_MIN;
    AV_STAT_R_MAX = AV_CHANNEL_STAT_MAX;
    AV_STAT_R_SUM = AV_CHANNEL_STAT_SUM;
    AV_STAT_G_MEAN = AV_CHANNEL_STAT_MEAN;
    AV_STAT_G_SD = AV_CHANNEL_STAT_SD;
    AV_STAT_G_MIN = AV_CHANNEL_STAT_MIN;
    AV_STAT_G_MAX = AV_CHANNEL_STAT_MAX;
    AV_STAT_G_SUM = AV_CHANNEL_STAT_SUM;
    AV_STAT_B_MEAN = AV_CHANNEL_STAT_MEAN;
    AV_STAT_B_SD = AV_CHANNEL_STAT_SD;
    AV_STAT_B_MIN = AV_CHANNEL_STAT_MIN;
    AV_STAT_B_MAX = AV_CHANNEL_STAT_MAX;
    AV_STAT_B_SUM = AV_CHANNEL_STAT_SUM;
}


function computeActiveChannelStatistics(left, top, diameter) {
    channelN = 0;
    channelSum = 0;
    channelSumSq = 0;
    channelMin = 1e30;
    channelMax = -1e30;
    radius = diameter/2;
    exactCenterX = left+radius;
    exactCenterY = top+radius;

    for (pixelY=top; pixelY<top+diameter; pixelY++) {
        for (pixelX=left; pixelX<left+diameter; pixelX++) {
            deltaX = pixelX+0.5-exactCenterX;
            deltaY = pixelY+0.5-exactCenterY;
            if ((deltaX*deltaX + deltaY*deltaY)>(radius*radius)) continue;
            if (pixelX<0) continue;
            if (pixelY<0) continue;
            if (pixelX>=AV_IMAGE_WIDTH) continue;
            if (pixelY>=AV_IMAGE_HEIGHT) continue;

            channelValue = getPixel(pixelX, pixelY);
            channelN++;
            channelSum += channelValue;
            channelSumSq += channelValue*channelValue;
            if (channelValue<channelMin) channelMin = channelValue;
            if (channelValue>channelMax) channelMax = channelValue;
        }
    }

    AV_CHANNEL_STAT_N = channelN;
    if (channelN==0) {
        AV_CHANNEL_STAT_MEAN = NaN;
        AV_CHANNEL_STAT_SD = NaN;
        AV_CHANNEL_STAT_MIN = NaN;
        AV_CHANNEL_STAT_MAX = NaN;
        AV_CHANNEL_STAT_SUM = NaN;
        return;
    }

    AV_CHANNEL_STAT_MEAN = channelSum/channelN;
    AV_CHANNEL_STAT_MIN = channelMin;
    AV_CHANNEL_STAT_MAX = channelMax;
    AV_CHANNEL_STAT_SUM = channelSum;
    if (channelN>1) {
        channelVariance = (channelSumSq-channelSum*channelSum/channelN)/(channelN-1);
        if (channelVariance<0) channelVariance = 0;
        AV_CHANNEL_STAT_SD = sqrt(channelVariance);
    } else {
        AV_CHANNEL_STAT_SD = 0;
    }
}


function computePackedRgbStatistics(left, top, diameter) {
    statN = 0;
    sumR = 0; sumG = 0; sumB = 0;
    sumSqR = 0; sumSqG = 0; sumSqB = 0;
    minR = 1e30; minG = 1e30; minB = 1e30;
    maxR = -1e30; maxG = -1e30; maxB = -1e30;
    radius = diameter/2;
    exactCenterX = left+radius;
    exactCenterY = top+radius;

    for (pixelY=top; pixelY<top+diameter; pixelY++) {
        for (pixelX=left; pixelX<left+diameter; pixelX++) {
            deltaX = pixelX+0.5-exactCenterX;
            deltaY = pixelY+0.5-exactCenterY;
            if ((deltaX*deltaX + deltaY*deltaY)>(radius*radius)) continue;

            // The full mathematical circle is retained in roi_x/roi_y/size,
            // but only the visible part has source pixels and can contribute
            // to intensity statistics.
            if (pixelX<0) continue;
            if (pixelY<0) continue;
            if (pixelX>=AV_IMAGE_WIDTH) continue;
            if (pixelY>=AV_IMAGE_HEIGHT) continue;

            rawPixel = getPixel(pixelX, pixelY);
            redValue = (rawPixel>>16)&255;
            greenValue = (rawPixel>>8)&255;
            blueValue = rawPixel&255;

            statN++;
            sumR += redValue; sumG += greenValue; sumB += blueValue;
            sumSqR += redValue*redValue;
            sumSqG += greenValue*greenValue;
            sumSqB += blueValue*blueValue;
            if (redValue<minR) minR = redValue;
            if (redValue>maxR) maxR = redValue;
            if (greenValue<minG) minG = greenValue;
            if (greenValue>maxG) maxG = greenValue;
            if (blueValue<minB) minB = blueValue;
            if (blueValue>maxB) maxB = blueValue;
        }
    }

    AV_STAT_N = statN;
    if (statN==0) {
        AV_STAT_R_MEAN=NaN; AV_STAT_R_SD=NaN; AV_STAT_R_MIN=NaN; AV_STAT_R_MAX=NaN; AV_STAT_R_SUM=NaN;
        AV_STAT_G_MEAN=NaN; AV_STAT_G_SD=NaN; AV_STAT_G_MIN=NaN; AV_STAT_G_MAX=NaN; AV_STAT_G_SUM=NaN;
        AV_STAT_B_MEAN=NaN; AV_STAT_B_SD=NaN; AV_STAT_B_MIN=NaN; AV_STAT_B_MAX=NaN; AV_STAT_B_SUM=NaN;
        return;
    }

    AV_STAT_R_MEAN = sumR/statN;
    AV_STAT_G_MEAN = sumG/statN;
    AV_STAT_B_MEAN = sumB/statN;
    AV_STAT_R_SUM = sumR;
    AV_STAT_G_SUM = sumG;
    AV_STAT_B_SUM = sumB;
    AV_STAT_R_MIN = minR; AV_STAT_R_MAX = maxR;
    AV_STAT_G_MIN = minG; AV_STAT_G_MAX = maxG;
    AV_STAT_B_MIN = minB; AV_STAT_B_MAX = maxB;

    if (statN>1) {
        varR = (sumSqR-sumR*sumR/statN)/(statN-1);
        varG = (sumSqG-sumG*sumG/statN)/(statN-1);
        varB = (sumSqB-sumB*sumB/statN)/(statN-1);
        if (varR<0) varR=0;
        if (varG<0) varG=0;
        if (varB<0) varB=0;
        AV_STAT_R_SD = sqrt(varR);
        AV_STAT_G_SD = sqrt(varG);
        AV_STAT_B_SD = sqrt(varB);
    } else {
        AV_STAT_R_SD = 0;
        AV_STAT_G_SD = 0;
        AV_STAT_B_SD = 0;
    }
}


function buildCombinedResults() {
    combinedText = completedCsvHeader();
    recordFiles = getFileList(AV_CSV_DIR);
    Array.sort(recordFiles);
    for (recordI=0; recordI<recordFiles.length; recordI++) {
        if (!endsWith(toLowerCase(recordFiles[recordI]), ".csv")) continue;
        recordText = replace(File.openAsString(AV_CSV_DIR+recordFiles[recordI]), "\r", "");
        recordLines = split(recordText, "\n");
        for (lineI=1; lineI<recordLines.length; lineI++)
            if (trim(recordLines[lineI])!="")
                combinedText = "" + combinedText + recordLines[lineI] + "\n";
    }
    File.saveString(combinedText, AV_OUTPUT_ROOT+"combined_results.csv");
    buildCombinedImageSummary();
}


function imageSummaryHeader() {
    return "image_name,image_relative_path,annotator,annotation_status,"+
        "cell_count,green_count,orange_count,image_width_px,image_height_px,"+
        "image_channels,image_bit_depth,roi_diameter_px,zoom_percent,"+
        "coverage_sectors_visited,coverage_sectors_total,macro_version,"+
        "summary_source,completed_at,uncertain_count\n";
}


function saveImageSummary(completedAt) {
    greenCount = countClass("green");
    orangeCount = countClass("orange");
    uncertainCount = countClass("uncertain");
    summaryText = imageSummaryHeader();
    summaryRow = "" + csvQuote(AV_IMAGE_NAME) + "," + csvQuote(AV_IMAGE_REL) + "," +
        csvQuote(AV_ANNOTATOR) + "," + csvQuote("complete") + "," +
        AV_COUNT + "," + greenCount + "," + orangeCount + "," +
        AV_IMAGE_WIDTH + "," + AV_IMAGE_HEIGHT + "," + AV_IMAGE_CHANNELS + "," +
        AV_IMAGE_BIT_DEPTH + "," + AV_DIAMETER + "," + AV_ACTUAL_ZOOM_PERCENT + "," +
        AV_COVERAGE_COUNT + ",9," + csvQuote(AV_VERSION) + "," +
        csvQuote("native_v"+AV_VERSION) + "," + csvQuote(completedAt) + "," +
        uncertainCount + "\n";
    summaryText = "" + summaryText + summaryRow;
    File.saveString(summaryText, AV_SUMMARY_PATH);
    if (!File.exists(AV_SUMMARY_PATH)) return 0;
    return 1;
}


function ensureCompletedImageSummaries() {
    for (legacyI=0; legacyI<AV_IMAGE_COUNT; legacyI++) {
        legacyCellPath = AV_CSV_DIR + AV_IMAGE_KEYS[legacyI] + ".csv";
        legacySummaryPath = AV_SUMMARY_DIR + AV_IMAGE_KEYS[legacyI] + ".csv";
        if (!File.exists(legacyCellPath)) continue;
        if (File.exists(legacySummaryPath)) continue;

        legacyText = replace(File.openAsString(legacyCellPath), "\r", "");
        legacyLines = split(legacyText, "\n");
        legacyCells = 0;
        legacyGreen = 0;
        legacyOrange = 0;
        legacyUncertain = 0;
        for (legacyLineI=1; legacyLineI<legacyLines.length; legacyLineI++) {
            legacyLine = trim(legacyLines[legacyLineI]);
            if (legacyLine=="") continue;
            legacyCells++;
            if (indexOf(legacyLine, ",\"green\",")>=0) legacyGreen++;
            if (indexOf(legacyLine, ",\"orange\",")>=0) legacyOrange++;
            if (indexOf(legacyLine, ",\"uncertain\",")>=0) legacyUncertain++;
        }

        // Metadata absent from a header-only v2.1.x CSV is intentionally left
        // blank instead of inventing an annotator, image geometry or timestamp.
        legacySummary = imageSummaryHeader();
        legacyRow = "" + csvQuote(File.getName(AV_IMAGE_PATHS[legacyI])) + "," +
            csvQuote(AV_IMAGE_RELS[legacyI]) + "," + csvQuote("") + "," +
            csvQuote("complete") + "," + legacyCells + "," + legacyGreen + "," +
            legacyOrange + ",,,,,,,,," + csvQuote("") + "," +
            csvQuote("legacy_csv_backfill") + "," + csvQuote("") + "," +
            legacyUncertain + "\n";
        legacySummary = "" + legacySummary + legacyRow;
        File.saveString(legacySummary, legacySummaryPath);
    }
}


function buildCombinedImageSummary() {
    summaryCombined = imageSummaryHeader();
    summaryFiles = getFileList(AV_SUMMARY_DIR);
    Array.sort(summaryFiles);
    for (summaryI=0; summaryI<summaryFiles.length; summaryI++) {
        if (!endsWith(toLowerCase(summaryFiles[summaryI]), ".csv")) continue;
        oneSummary = replace(File.openAsString(AV_SUMMARY_DIR+summaryFiles[summaryI]), "\r", "");
        summaryLines = split(oneSummary, "\n");
        for (summaryLineI=1; summaryLineI<summaryLines.length; summaryLineI++)
            if (trim(summaryLines[summaryLineI])!="")
                summaryCombined = "" + summaryCombined + summaryLines[summaryLineI] + "\n";
    }
    File.saveString(summaryCombined, AV_OUTPUT_ROOT+"combined_image_summary.csv");
}


function makeDirectoryIfNeeded(path) {
    if (!File.exists(path)) File.makeDirectory(path);
    if (!File.exists(path)) exit("Could not create output folder:\n" + path);
}


function csvQuote(value) {
    quote = fromCharCode(34);
    value = ""+value;
    value = replace(value, quote, quote+quote);
    return quote+value+quote;
}


function numberText(value) {
    if (isNaN(value)) return "";
    return d2s(value, 4);
}


function timeStamp() {
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
    // Starting this expression with a number makes ImageJ 1.54p infer a
    // numeric function return, then fail when the text timestamp is returned.
    stamp = ""+year+"-"+pad2(month+1)+"-"+pad2(dayOfMonth)+" "+
        pad2(hour)+":"+pad2(minute)+":"+pad2(second);
    return stamp;
}


function pad2(value) {
    value = round(value);
    if (value<10) return "0"+value;
    return ""+value;
}
