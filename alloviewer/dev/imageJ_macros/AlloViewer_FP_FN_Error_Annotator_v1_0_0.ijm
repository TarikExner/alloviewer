/*
 * AlloViewer FP/FN Error Annotator
 * Version 1.0.0 (2026-11-14)
 *
 * Manual review of AlloViewer classification errors on images that already
 * contain the pipeline's segmentation/call outlines. The source image is
 * never changed. This macro records only unambiguous classification errors:
 *
 *   FP = MAGENTA pipeline outline (positive/orange call), but the cell is
 *        truly negative/green
 *   FN = YELLOW pipeline outline (negative/green call), but the cell is
 *        truly positive/orange
 *
 * Ignore cyan uncertain calls, missed/unsegmented cells, and mixed-truth
 * merged clusters for which one pipeline object is only partly correct.
 *
 * CONTROLS WHILE THE IMAGE WINDOW IS ACTIVE
 * -----------------------------------------
 *   Click or drag-release          add FP circle (RED)
 *   Shift + click/drag-release     add FN circle (WHITE)
 *   Option/Alt + click             undo the most recent circle
 *   Hold Space + left-drag         pan the zoomed image
 *   Hold Shift+Option+Space (0.6s) complete image and open next
 *   Hold Option + Space (0.6s)     save draft and pause
 *   Space tap / Shift + Space      no action (safety lock)
 *
 * Each edit is saved immediately as a draft. Completed images receive a
 * per-image error CSV and a per-image summary, including images with zero
 * errors. Circular ROIs may extend beyond an image boundary. Images open
 * maximized at 300% zoom by default. A 6 x 5 coverage reminder reduces the
 * risk of skipping parts of these full-size images.
 */

var AV_VERSION = "1.0.0";
var AV_MAX_IMAGES = 100000;
var AV_MAX_CELLS = 10000;
var AV_COMMAND_HOLD_MS = 600;
var AV_COVERAGE_COLS = 6;
var AV_COVERAGE_ROWS = 5;
var AV_COVERAGE_TOTAL = 30;
var AV_FP_COLOR = "ff0000";
var AV_FN_COLOR = "ffffff";
var AV_STROKE_WIDTH = 2.0;

var AV_IMAGE_ROOT = "";
var AV_OUTPUT_ROOT = "";
var AV_CSV_DIR = "";
var AV_DRAFT_DIR = "";
var AV_SUMMARY_DIR = "";
var AV_ANNOTATOR = "annotator";
var AV_DIAMETER = 8;
var AV_TARGET_ZOOM_PERCENT = 300;
var AV_ACTUAL_ZOOM_PERCENT = 300;

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

macro "Run AlloViewer FP/FN Error Annotator v1.0.0" {
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

    Dialog.create("AlloViewer FP/FN error review");
    Dialog.addString("Annotator name or ID:", "annotator", 24);
    Dialog.addNumber("Annotation zoom (percent):", 300);
    Dialog.addMessage(
        "FP: magenta outline (positive/orange call), but truly negative/green.\n"+
        "FN: yellow outline (negative/green call), but truly positive/orange.\n"+
        "Ignore cyan uncertain calls, unsegmented cells and mixed-truth merged clusters.\n\n"+
        "The circle diameter is fixed at 8 pixels.\n"+
        "Recommended zoom for these full-size overlays: 300%.\n"+
        "Subfolders are searched, so one shared parent folder can contain both datasets.\n"+
        "If several annotators review independently, give each one a separate source-folder copy."
    );
    Dialog.show();
    AV_ANNOTATOR = Dialog.getString();
    AV_TARGET_ZOOM_PERCENT = round(Dialog.getNumber());
    if (AV_ANNOTATOR=="") AV_ANNOTATOR = "annotator";
    AV_DIAMETER = 8;
    if (AV_TARGET_ZOOM_PERCENT<100) AV_TARGET_ZOOM_PERCENT = 300;
    if (AV_TARGET_ZOOM_PERCENT>3200) AV_TARGET_ZOOM_PERCENT = 3200;

    AV_OUTPUT_ROOT = AV_IMAGE_ROOT + "AlloViewer_FP_FN_review" + File.separator;
    AV_CSV_DIR = AV_OUTPUT_ROOT + "per_image_csv" + File.separator;
    AV_DRAFT_DIR = AV_OUTPUT_ROOT + "drafts" + File.separator;
    AV_SUMMARY_DIR = AV_OUTPUT_ROOT + "per_image_summary" + File.separator;
    makeDirectoryIfNeeded(AV_OUTPUT_ROOT);
    makeDirectoryIfNeeded(AV_CSV_DIR);
    makeDirectoryIfNeeded(AV_DRAFT_DIR);
    makeDirectoryIfNeeded(AV_SUMMARY_DIR);

    completed = countCompletedImages();
    remaining = AV_IMAGE_COUNT-completed;

    showMessage("AlloViewer FP/FN error annotator",
        "Images found: " + AV_IMAGE_COUNT +
        "\nAlready complete: " + completed +
        "\nRemaining: " + remaining +
        "\n\nClick or drag-release: FP (red circle)"+
        "\nShift + click/drag-release: FN (white circle)"+
        "\nOption/Alt + click: undo"+
        "\nHold Space + left-drag: pan at high zoom"+
        "\nHold Shift + Option + Space for 0.6 s: complete image and open next"+
        "\nHold Option + Space for 0.6 s: save draft and pause"+
        "\nSpace tap or Shift + Space: no action (safety lock)"+
        "\n\nCircular ROI diameter: " + AV_DIAMETER + " px"+
        "\nAnnotation zoom target: " + AV_TARGET_ZOOM_PERCENT + "%"+
        "\nA first held Shift + Option + Space warns if all 30 review sectors were not visited."+
        "\nHold Shift + Option + Space again to finish anyway."+
        "\n\nFP means magenta pipeline outline but truly negative/green."+
        "\nFN means yellow pipeline outline but truly positive/orange."+
        "\nDo not mark cyan uncertain calls, unsegmented cells, or mixed-truth merged clusters."+
        "\nThe source image pixels are never changed.");

    while (1) {
        nextIndex = findNextImageIndex();
        if (nextIndex<0) {
            buildCombinedResults();
            showMessage("AlloViewer: FP/FN review complete",
                "All " + AV_IMAGE_COUNT + " discovered images are complete.\n\n"+
                "Per-image CSV files:\n" + AV_CSV_DIR +
                "\n\nCombined FP/FN CSV:\n" + AV_OUTPUT_ROOT + "combined_fp_fn_errors.csv"+
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
            if (entry=="AlloViewer_FP_FN_review/")
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
    pressAction = "FP";
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

                if (AV_COVERAGE_COUNT<AV_COVERAGE_TOTAL && !AV_FINISH_ARMED) {
                    beep();
                    AV_FINISH_ARMED = 1;
                    showStatus("!Coverage " + AV_COVERAGE_COUNT + "/" + AV_COVERAGE_TOTAL +
                        ". Review the remaining sectors, or hold Shift + Option + Space again for 0.6 s to finish anyway.");
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
            if ((flags&8)!=0)
                pressAction = "undo";
            else if ((flags&1)!=0)
                pressAction = "FN";
            else
                pressAction = "FP";
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
    if (cellClass=="FP")
        Roi.setStrokeColor(AV_FP_COLOR);
    else
        Roi.setStrokeColor(AV_FN_COLOR);
    Roi.setStrokeWidth(AV_STROKE_WIDTH);
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
    if (cellClass=="FP")
        Overlay.addSelection(AV_FP_COLOR, AV_STROKE_WIDTH);
    else
        Overlay.addSelection(AV_FN_COLOR, AV_STROKE_WIDTH);
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
        if (AV_CLASS[drawI]=="FP")
            Overlay.addSelection(AV_FP_COLOR, AV_STROKE_WIDTH);
        else
            Overlay.addSelection(AV_FN_COLOR, AV_STROKE_WIDTH);
    }
    Overlay.drawLabels(false);
    Overlay.selectable(false);
    Overlay.show;
    run("Select None");
}


function updateStatusLine() {
    fpCount = countClass("FP");
    fnCount = countClass("FN");
    if (AV_FINISH_ARMED) {
        showStatus("!Coverage " + AV_COVERAGE_COUNT + "/" + AV_COVERAGE_TOTAL +
            ". Review remaining sectors, or hold Shift + Option + Space again for 0.6 s to finish anyway.");
        return;
    }
    showStatus("!" + AV_IMAGE_REL + " | FP=" + fpCount +
        " FN=" + fnCount +
        " | circle=" + AV_DIAMETER +
        "px | coverage=" + AV_COVERAGE_COUNT + "/" + AV_COVERAGE_TOTAL +
        " | click=FP, Shift-click=FN, Option-click=undo, Space-drag=pan, hold Shift-Option-Space=next");
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

    coverageColumn = floor(AV_COVERAGE_COLS*cursorX/AV_IMAGE_WIDTH);
    coverageRow = floor(AV_COVERAGE_ROWS*cursorY/AV_IMAGE_HEIGHT);
    if (coverageColumn>=AV_COVERAGE_COLS) coverageColumn = AV_COVERAGE_COLS-1;
    if (coverageRow>=AV_COVERAGE_ROWS) coverageRow = AV_COVERAGE_ROWS-1;
    coverageIndex = AV_COVERAGE_COLS*coverageRow+coverageColumn;
    if (AV_COVERAGE[coverageIndex]==0) {
        AV_COVERAGE[coverageIndex] = 1;
        AV_COVERAGE_COUNT++;
        return 1;
    }
    return 0;
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
    // the requested magnification is reached. At 300%, an 8 px circle is about
    // 24 screen pixels across, while ImageJ's Space-drag pans the canvas.
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
    AV_COVERAGE = newArray(AV_COVERAGE_TOTAL);
    for (resetI=0; resetI<AV_COVERAGE_TOTAL; resetI++) AV_COVERAGE[resetI] = 0;
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
        if (draftClass!="FP" && draftClass!="FN") continue;
        AV_X[AV_COUNT] = draftX;
        AV_Y[AV_COUNT] = draftY;
        AV_SIZE[AV_COUNT] = draftSize;
        AV_CLASS[AV_COUNT] = draftClass;
        AV_COUNT++;
    }
}


function completedCsvHeader() {
    return "image_name,image_relative_path,error_id,error_type,pipeline_call,"+
        "reference_call,annotator,roi_shape,center_x_px,center_y_px,roi_x_px,"+
        "roi_y_px,diameter_px,radius_px,mask_pixel_count,image_width_px,"+
        "image_height_px,image_channels,image_bit_depth,completed_at,macro_version\n";
}


function saveCompletedAnnotation() {
    completedAt = timeStamp();
    csvText = completedCsvHeader();

    for (saveI=0; saveI<AV_COUNT; saveI++) {
        centerX = AV_X[saveI] + AV_SIZE[saveI]/2;
        centerY = AV_Y[saveI] + AV_SIZE[saveI]/2;
        radius = AV_SIZE[saveI]/2;
        maskPixelCount = visibleCirclePixelCount(
            AV_X[saveI], AV_Y[saveI], AV_SIZE[saveI]);

        if (AV_CLASS[saveI]=="FP") {
            pipelineCall = "positive";
            referenceCall = "negative";
        } else {
            pipelineCall = "negative";
            referenceCall = "positive";
        }

        // The explicit empty-string prefix is required by ImageJ 1.54p. With
        // a string-returning user function as the first term of a += chain,
        // that interpreter can incorrectly demand a numeric return value.
        rowText = "" + csvQuote(AV_IMAGE_NAME) + "," + csvQuote(AV_IMAGE_REL) + "," +
            (saveI+1) + "," + csvQuote(AV_CLASS[saveI]) + "," +
            csvQuote(pipelineCall) + "," + csvQuote(referenceCall) + "," +
            csvQuote(AV_ANNOTATOR) + "," + csvQuote("circle") + "," +
            numberText(centerX) + "," + numberText(centerY) + "," +
            AV_X[saveI] + "," + AV_Y[saveI] + "," + AV_SIZE[saveI] + "," +
            numberText(radius) + "," + maskPixelCount + "," + AV_IMAGE_WIDTH + "," +
            AV_IMAGE_HEIGHT + "," + AV_IMAGE_CHANNELS + "," + AV_IMAGE_BIT_DEPTH + "," +
            csvQuote(completedAt) + "," + csvQuote(AV_VERSION) + "\n";
        csvText = "" + csvText + rowText;
    }

    // Write the image-level row first. The per-image error file remains the
    // completion marker, so a failed summary write cannot silently mark done.
    if (!saveImageSummary(completedAt)) return 0;
    File.saveString(csvText, AV_FINAL_PATH);
    if (!File.exists(AV_FINAL_PATH)) return 0;
    if (File.exists(AV_DRAFT_PATH)) File.delete(AV_DRAFT_PATH);
    return 1;
}


function visibleCirclePixelCount(left, top, diameter) {
    visibleCount = 0;
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
            visibleCount++;
        }
    }
    return visibleCount;
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
    File.saveString(combinedText, AV_OUTPUT_ROOT+"combined_fp_fn_errors.csv");
    buildCombinedImageSummary();
}


function imageSummaryHeader() {
    return "image_name,image_relative_path,annotator,annotation_status,"+
        "error_count,fp_count,fn_count,image_width_px,image_height_px,"+
        "image_channels,image_bit_depth,roi_diameter_px,zoom_percent,"+
        "coverage_sectors_visited,coverage_sectors_total,macro_version,"+
        "summary_source,completed_at\n";
}


function saveImageSummary(completedAt) {
    fpCount = countClass("FP");
    fnCount = countClass("FN");
    summaryText = imageSummaryHeader();
    summaryRow = "" + csvQuote(AV_IMAGE_NAME) + "," + csvQuote(AV_IMAGE_REL) + "," +
        csvQuote(AV_ANNOTATOR) + "," + csvQuote("complete") + "," +
        AV_COUNT + "," + fpCount + "," + fnCount + "," +
        AV_IMAGE_WIDTH + "," + AV_IMAGE_HEIGHT + "," + AV_IMAGE_CHANNELS + "," +
        AV_IMAGE_BIT_DEPTH + "," + AV_DIAMETER + "," + AV_ACTUAL_ZOOM_PERCENT + "," +
        AV_COVERAGE_COUNT + "," + AV_COVERAGE_TOTAL + "," + csvQuote(AV_VERSION) + "," +
        csvQuote("native_v"+AV_VERSION) + "," + csvQuote(completedAt) + "\n";
    summaryText = "" + summaryText + summaryRow;
    File.saveString(summaryText, AV_SUMMARY_PATH);
    if (!File.exists(AV_SUMMARY_PATH)) return 0;
    return 1;
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

