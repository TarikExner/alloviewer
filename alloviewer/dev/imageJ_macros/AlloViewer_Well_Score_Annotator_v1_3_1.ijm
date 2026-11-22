/*
 * AlloViewer Well Score Annotator
 * Version 1.3.1 (2026-10-26)
 *
 * PURPOSE
 * -------
 * Score one whole-well TIFF at a time with exactly one ordinal value:
 * 1, 2, 4, 6, 8 or 0.
 *
 * HOW TO START
 * ------------
 * Drag this .ijm file onto Fiji and click Run in the Script Editor.
 * This is a run-once macro. It installs no toolbar tools and makes no
 * permanent changes to Fiji keyboard shortcuts.
 *
 * WORKFLOW
 * --------
 * 1. Choose whether to continue normally, score one specifically selected
 *    incomplete assay, or rebuild the combined CSV from final assay CSVs.
 * 2. Enter a stable annotator ID and choose the assigned annotation pass.
 * 3. Choose one shared result-folder parent used by every annotator.
 * 4. Fiji displays the exact assay name and asks for that assay folder.
 * 5. The macro accepts only a folder with the exact assay name and the exact
 *    expected number of numeric Bild_XXXX.tif files.
 * 6. Images are sorted by the numeric XXXX value and mapped to the fixed
 *    60-well vertical-snake order embedded below.
 * 7. Press 1, 2, 4, 6, 8 or 0 while the scoring image is active. The large
 *    score buttons at the right remain available as a mouse fallback.
 *    One key release or click records the score, saves the draft and advances.
 *
 * PANEL CONTROLS
 * --------------
 * KEYS 1 / 2 / 4 / 6 / 8 / 0    score and advance
 * SCORE 1 / 2 / 4 / 6 / 8 / 0   mouse fallback: score and advance
 * BACK                           reopen the previous well for correction
 * PAUSE                          save draft, close image and stop safely
 *
 * SAFETY
 * ------
 * - Original images and the supplied Frankfurt/Berlin result sheets are
 *   never written to.
 * - A deterministic final CSV is created once per assay and annotation slot.
 *   Existing final CSVs are skipped and never overwritten.
 * - Progress is determined only from per-assay final CSVs. The combined CSV
 *   is never treated as a progress marker and can be rebuilt at any time.
 * - Final and draft filenames deliberately retain the existing "__0" key
 *   used by Fiji/ImageJ 1.54p. This makes earlier annotations resumable and
 *   avoids the version-dependent IJ.checksum result that hid valid files.
 * - Every click is immediately checkpointed to a draft.
 * - Every image window created while preparing a well is tracked and closed
 *   after scoring, BACK, PAUSE or an opening error. Images that were already
 *   open before that well are never closed by this cleanup.
 * - For each center, all macro-created annotation slots reject the same
 *   annotator ID for the same assay, including unfinished drafts.
 * - Every final row records well, image filename, numeric image number,
 *   file size, file modification time, score and timestamps.
 * - Display uses the fixed 0-1023 acquisition range for these 10-bit TIFFs;
 *   there is no per-image auto-normalization that could bias scoring.
 */

var AV_VERSION = "1.3.1";
var AV_DISPLAY_MIN = 0;
var AV_DISPLAY_MAX = 1023;
var AV_MAX_IMAGES = 1000;

var AV_WELLS = newArray(
    "A1", "B1", "C1", "D1", "E1", "F1",
    "F2", "E2", "D2", "C2", "B2", "A2",
    "A3", "B3", "C3", "D3", "E3", "F3",
    "F4", "E4", "D4", "C4", "B4", "A4",
    "A5", "B5", "C5", "D5", "E5", "F5",
    "F6", "E6", "D6", "C6", "B6", "A6",
    "A7", "B7", "C7", "D7", "E7", "F7",
    "F8", "E8", "D8", "C8", "B8", "A8",
    "A9", "B9", "C9", "D9", "E9", "F9",
    "F10", "E10", "D10", "C10", "B10", "A10"
);

var AV_CENTER = "";
var AV_SLOT = 0;
var AV_RUN_ACTION = "";
var AV_PASS_LABEL = "";
var AV_ANNOTATOR = "";
var AV_OUTPUT_PARENT = "";
var AV_PROJECT_DIR = "";
var AV_FINAL_DIR = "";
var AV_DRAFT_DIR = "";
var AV_COMBINED_DIR = "";
var AV_OTHER_FINAL_DIRS = newArray(0);
var AV_OTHER_DRAFT_DIRS = newArray(0);
var AV_OTHER_SLOT_COUNT = 0;
var AV_LAST_COMBINED_ASSAYS = 0;
var AV_LAST_COMBINED_ROWS = 0;
var AV_LAST_COMBINED_OK = 0;

var AV_ASSAY_NAMES = newArray(0);
var AV_ASSAY_EXPECTED = newArray(0);
var AV_ASSAY_TYPE = newArray(0);
var AV_ASSAY_PRA = newArray(0);
var AV_ASSAY_SPECIFICITY = newArray(0);
var AV_ASSAY_COUNT = 0;
var AV_SESSION_SKIP = newArray(0);

var AV_CURRENT_ASSAY_INDEX = -1;
var AV_CURRENT_ASSAY = "";
var AV_CURRENT_EXPECTED = 0;
var AV_CURRENT_TYPE = "";
var AV_CURRENT_PRA = "";
var AV_CURRENT_SPECIFICITY = "";
var AV_ASSAY_KEY = "";
var AV_SOURCE_DIR = "";
var AV_FINAL_PATH = "";
var AV_DRAFT_PATH = "";

var AV_IMAGE_NAMES = newArray(0);
var AV_IMAGE_PATHS = newArray(0);
var AV_IMAGE_NUMBERS = newArray(0);
var AV_IMAGE_BYTES = newArray(0);
var AV_IMAGE_MODIFIED = newArray(0);
var AV_IMAGE_COUNT = 0;
var AV_GAP_COUNT = 0;
var AV_GAP_TEXT = "";
var AV_SCAN_ERROR = "";

var AV_SCORES = newArray(0);
var AV_SCORE_SET = newArray(0);
var AV_SCORE_TIMES = newArray(0);
var AV_NEXT_INDEX = 0;
var AV_DRAFT_ERROR = "";

var AV_IMAGE_ID = 0;
var AV_IMAGE_IDS_BEFORE = "";
var AV_OWNED_IMAGE_IDS = "";
var AV_SOURCE_WIDTH = 0;
var AV_SOURCE_HEIGHT = 0;
var AV_SOURCE_CHANNELS = 0;
var AV_PANEL_WIDTH = 0;
var AV_PANEL_HEADER = 0;
var AV_BUTTON_HEIGHT = 0;
var AV_CANVAS_WIDTH = 0;
var AV_REFERENCE_WIDTH = 0;
var AV_REFERENCE_HEIGHT = 0;
var AV_OPEN_ERROR = "";
var AV_KEY_PROPERTY = "AlloViewer.WellScore.Action";
var AV_KEY_HOLDER = "AlloViewer.WellScore.Dispatcher";
var AV_KEY_CAPTURE_OK = 0;
var AV_KEY_WARNING_SHOWN = 0;


if (getArgument()=="SELFTEST")
    runSelfTests();
else if (getArgument()=="KEYSELFTEST")
    runKeyboardBridgeSelfTest();
else
    runWellScoreAnnotator();


function runWellScoreAnnotator() {
    requires("1.54i");

    actionChoices = newArray(
        "Continue all eligible incomplete assays",
        "Choose one incomplete assay",
        "Rebuild combined CSV from final assay files only"
    );

    passChoices = newArray(
        "Frankfurt - annotator slot 3 (43 assay folders)",
        "Frankfurt - annotator slot 4 (43 assay folders)",
        "Berlin - annotator slot 2 (10 assay folders)",
        "Berlin - annotator slot 3 (10 assay folders)",
        "Berlin - annotator slot 4 (10 assay folders)"
    );

    Dialog.create("AlloViewer well scoring");
    Dialog.addChoice("Action:", actionChoices, actionChoices[0]);
    Dialog.addChoice("Assigned annotation pass:", passChoices, passChoices[0]);
    Dialog.addString("Annotator name or stable ID:", "", 28);
    Dialog.addDirectory("Shared result-folder parent:", getDirectory("home"));
    Dialog.addMessage(
        "Use the same shared result folder for every annotator.\n"+
        "Do not use another slot for the same center under the same annotator ID.\n\n"+
        "The macro never edits the supplied annotation sheets or source images."
    );
    Dialog.show();

    AV_RUN_ACTION = Dialog.getChoice();
    AV_PASS_LABEL = Dialog.getChoice();
    AV_ANNOTATOR = trim(Dialog.getString());
    AV_OUTPUT_PARENT = withTrailingSeparator(Dialog.getString());

    if (!validAnnotatorId(AV_ANNOTATOR)) {
        showMessage("AlloViewer well scoring",
            "Enter a stable annotator ID using letters, numbers, spaces, periods, hyphens or underscores only.\n\n"+
            "Do not use commas, tabs or changing nicknames; the ID is used to keep annotation slots independent.");
        return;
    }
    if (AV_OUTPUT_PARENT=="") {
        showMessage("AlloViewer well scoring", "No result folder was selected.");
        return;
    }

    configurePass(AV_PASS_LABEL);
    configureOutputFolders();

    if (startsWith(AV_RUN_ACTION, "Rebuild combined")) {
        combinedOk = buildCombinedPassFile();
        if (combinedOk) {
            showMessage("Combined CSV rebuilt",
                "The combined file was recreated exclusively from the per-assay final CSVs.\n\n"+
                "Included final assays: " + AV_LAST_COMBINED_ASSAYS +
                "\nIncluded well rows: " + AV_LAST_COMBINED_ROWS +
                "\n\nOutput:\n" + combinedPassPath());
        } else {
            showMessage("Combined CSV could not be rebuilt",
                "The per-assay final CSVs were not changed.\n\n"+
                "Close the existing combined CSV if it is open in Excel and run this action again.\n\n"+
                "Expected output:\n" + combinedPassPath());
        }
        return;
    }

    singleAssayMode = startsWith(AV_RUN_ACTION, "Choose one incomplete");
    selectedAssay = -1;
    if (singleAssayMode) {
        selectedAssay = chooseOneIncompleteAssay();
        if (selectedAssay<0) {
            buildCombinedPassFile();
            showMessage("No eligible incomplete assay",
                "No incomplete assay is available for this annotator ID and annotation slot.\n\n"+
                "The combined CSV was rebuilt from the existing final assay files.");
            return;
        }
    }

    completedAtStart = countCompletedAssays();
    if (singleAssayMode)
        startDetail = "\nSelected assay: " + AV_ASSAY_NAMES[selectedAssay];
    else
        startDetail = "\n\nFor every remaining assay, Fiji will show the exact folder name before opening the folder chooser.";
    showMessage("AlloViewer well scoring",
        "Assigned pass: " + AV_CENTER + " annotator slot " + AV_SLOT +
        "\nAnnotator ID: " + AV_ANNOTATOR +
        "\nAssay folders in this pass: " + AV_ASSAY_COUNT +
        "\nAlready complete: " + completedAtStart +
        "\nRemaining without a final CSV: " + (AV_ASSAY_COUNT-completedAtStart) +
        "\nFinal-file directory: " + AV_FINAL_DIR +
        startDetail +
        "\n\nPress 1, 2, 4, 6, 8 or 0 to save and advance. The right-hand buttons remain available as a mouse fallback. Use BACK to correct the previous well or PAUSE to stop safely."
    );

    while (1) {
        if (singleAssayMode)
            nextAssay = selectedAssay;
        else
            nextAssay = findNextEligibleAssay();
        if (nextAssay<0) {
            buildCombinedPassFile();
            completedNow = countCompletedAssays();
            remainingNow = AV_ASSAY_COUNT-completedNow;
            if (remainingNow==0) {
                showMessage("AlloViewer pass complete",
                    "All " + AV_ASSAY_COUNT + " assays are complete for:\n"+
                    AV_CENTER + " annotator slot " + AV_SLOT +
                    "\n\nCombined long-format CSV:\n" + combinedPassPath());
            } else {
                showMessage("AlloViewer: no eligible assay",
                    "Complete files: " + completedNow + " of " + AV_ASSAY_COUNT +
                    "\nRemaining files: " + remainingNow +
                    "\n\nThe remaining assays are either skipped for this session, claimed by a draft from another annotator, or already scored/drafted by this same annotator in another slot for this center.\n\n"+
                    "Use the other independent annotator, or ask the project lead to review the draft files. No final file was changed.");
            }
            return;
        }

        setCurrentAssay(nextAssay);
        chosen = chooseValidatedAssayFolder();
        if (chosen=="") {
            if (singleAssayMode) {
                buildCombinedPassFile();
                showMessage("Selected assay not scored",
                    "No final CSV was created for:\n" + AV_CURRENT_ASSAY +
                    "\n\nThe combined CSV was rebuilt from the existing final assay files only.");
                return;
            }
            AV_SESSION_SKIP[nextAssay] = 1;
            continue;
        }
        AV_SOURCE_DIR = chosen;

        resetScoreArrays();
        draftState = loadDraft();
        if (draftState<0) {
            showMessage("AlloViewer: draft cannot be resumed", AV_DRAFT_ERROR);
            if (singleAssayMode) return;
            AV_SESSION_SKIP[nextAssay] = 1;
            continue;
        }

        scoreAction = scoreCurrentAssay();
        if (scoreAction=="pause") {
            buildCombinedPassFile();
            showMessage("AlloViewer paused",
                "The current assay is saved as a draft.\n\nAssay: " + AV_CURRENT_ASSAY +
                "\nNext well: " + nextWellText() +
                "\n\nRun the macro again with the same pass, annotator ID and shared result folder to resume.");
            return;
        }
        if (scoreAction=="error") {
            buildCombinedPassFile();
            showMessage("AlloViewer paused after an error",
                AV_OPEN_ERROR + "\n\nThe draft was retained and no final CSV was created for this assay.");
            return;
        }

        if (!saveFinalAssay()) {
            saveDraft();
            buildCombinedPassFile();
            showMessage("AlloViewer save error",
                "The final assay CSV could not be created safely.\n"+
                "The draft was retained and no existing final file was overwritten.\n\n"+
                "Expected final path:\n" + AV_FINAL_PATH);
            return;
        }
        combinedOk = buildCombinedPassFile();
        if (singleAssayMode) {
            if (combinedOk) {
                showMessage("Selected assay complete",
                    "Final CSV created for:\n" + AV_CURRENT_ASSAY +
                    "\n\nThe combined CSV was rebuilt from " + AV_LAST_COMBINED_ASSAYS +
                    " final assay files and contains " + AV_LAST_COMBINED_ROWS + " well rows.\n\n"+
                    "Combined output:\n" + combinedPassPath());
            } else {
                showMessage("Assay saved; combined rebuild failed",
                    "The per-assay final CSV was created safely for:\n" + AV_CURRENT_ASSAY +
                    "\n\nClose the existing combined CSV if it is open in Excel, then run the macro with the rebuild-only action.");
            }
            return;
        }
    }
}


function configurePass(passLabel) {
    if (indexOf(passLabel, "slot 2")>=0)
        AV_SLOT = 2;
    else if (indexOf(passLabel, "slot 4")>=0)
        AV_SLOT = 4;
    else
        AV_SLOT = 3;

    if (startsWith(passLabel, "Frankfurt")) {
        AV_CENTER = "Frankfurt";
        loadFrankfurtManifest();
    } else {
        AV_CENTER = "Berlin";
        loadBerlinManifest();
    }
    AV_SESSION_SKIP = newArray(AV_ASSAY_COUNT);
}


function loadFrankfurtManifest() {
    AV_ASSAY_COUNT = 43;
    AV_ASSAY_NAMES = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_EXPECTED = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_TYPE = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_PRA = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_SPECIFICITY = newArray(AV_ASSAY_COUNT);

    setAssay(0,  "20251024_25720349",          60, "CDC-PRA", "100", "poly");
    setAssay(1,  "20251028_25720349_+DTT",     60, "CDC-PRA", "61", "poly");
    setAssay(2,  "20251029_25720760",          60, "CDC-PRA", "0", "negative");
    setAssay(3,  "20251029_25721061",          60, "CDC-PRA", "0", "negative");
    setAssay(4,  "20251029_25721150",          60, "CDC-PRA", "0", "negative");
    setAssay(5,  "20251029_25721153",          60, "CDC-PRA", "0", "negative");
    setAssay(6,  "20251029_25721155",          60, "CDC-PRA", "0", "negative");
    setAssay(7,  "20251024_25720636",          60, "CDC-PRA", "50", "unspecific");
    setAssay(8,  "20251024_25720763",          60, "CDC-PRA", "29", "unspecific");
    setAssay(9,  "20251030_25065187",          60, "CDC-PRA", "0", "negative");
    setAssay(10, "20251030_25065181",          60, "CDC-PRA", "13", "unspecific");
    setAssay(11, "20251031_25720640",          60, "CDC-PRA", "20", "unspecific");
    setAssay(12, "20251125_25723077_+DTT",     60, "CDC-PRA", "", "unspecific");
    setAssay(13, "20251021_24702439",          60, "CDC-PRA", "12", "A1");
    setAssay(14, "20251021_25720330",          60, "CDC-PRA", "50", "poly");
    setAssay(15, "20251022_25720330_+DTT",     60, "CDC-PRA", "0", "negative");
    // The source sheet contains this physical folder twice, once with PRA 5
    // and once with PRA 3. It is scored once; both values are retained here
    // so the later merge must resolve the duplicated source records explicitly.
    setAssay(16, "20251021_25720338",          60, "CDC-PRA", "5|3", "negative");
    // Existing source rows mark column 10 (F10 through A10) as NA for both
    // CDC-XM folders, so these two assays intentionally contain 54 images.
    setAssay(17, "20251103_ET173293",          54, "CDC-XM", "NA", "NA");
    setAssay(18, "20251103_ET173293_+DTT",     54, "CDC-XM", "NA", "NA");
    setAssay(19, "20251106_25722169",          60, "CDC-PRA", "0", "negative");
    setAssay(20, "20251106_25722269",          60, "CDC-PRA", "0", "negative");
    setAssay(21, "20251107_25722332",          60, "CDC-PRA", "0", "negative");
    setAssay(22, "20251030_25720847",          60, "CDC-PRA", "0", "negative");
    setAssay(23, "20251030_25721169",          60, "CDC-PRA", "0", "negative");
    setAssay(24, "20251202_24702439_charge",   60, "CDC-PRA", "NA", "NA");
    setAssay(25, "20251202_24702439_shipment", 60, "CDC-PRA", "NA", "NA");
    setAssay(26, "20251030_25720841",          60, "CDC-PRA", "0", "negative");
    setAssay(27, "20251014_25719852",          60, "CDC-PRA", "0", "negative");
    setAssay(28, "20251014_25720237",          60, "CDC-PRA", "0", "negative");
    setAssay(29, "20251014_25720084",          60, "CDC-PRA", "0", "negative");
    setAssay(30, "20251014_25719960",          60, "CDC-PRA", "0", "negative");
    setAssay(31, "20251022_25720450",          60, "CDC-PRA", "0", "negative");
    setAssay(32, "20251022_25720531",          60, "CDC-PRA", "0", "negative");
    setAssay(33, "20251021_25720449",          60, "CDC-PRA", "0", "negative");
    setAssay(34, "20251024_25720508",          60, "CDC-PRA", "0", "negative");
    setAssay(35, "20251024_25720334",          60, "CDC-PRA", "0", "negative");
    setAssay(36, "20251021_25720345",          60, "CDC-PRA", "0", "negative");
    setAssay(37, "20251106_25065441",          60, "CDC-PRA", "0", "negative");
    setAssay(38, "20251028_25720445",          60, "CDC-PRA", "0", "negative");
    setAssay(39, "20251107_25065521",          60, "CDC-PRA", "0", "negative");
    setAssay(40, "20251203_24702438_charge",   60, "CDC-PRA", "NA", "NA");
    setAssay(41, "20251204_25713973_shipment", 60, "CDC-PRA", "NA", "NA");
    setAssay(42, "20251204_25713976_shipment", 60, "CDC-PRA", "NA", "NA");
}


function loadBerlinManifest() {
    AV_ASSAY_COUNT = 10;
    AV_ASSAY_NAMES = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_EXPECTED = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_TYPE = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_PRA = newArray(AV_ASSAY_COUNT);
    AV_ASSAY_SPECIFICITY = newArray(AV_ASSAY_COUNT);

    setAssay(0, "20260811_01",      60, "CDC-PRA", "87.5", "poly");
    setAssay(1, "20260811_01_+DTT", 60, "CDC-PRA", "94.64", "poly");
    setAssay(2, "20260811_02",      60, "CDC-PRA", "28.57", "poly");
    setAssay(3, "20260811_02_+DTT", 60, "CDC-PRA", "7.14", "poly");
    setAssay(4, "20260811_03",      60, "CDC-PRA", "32.14", "poly");
    setAssay(5, "20260811_04",      60, "CDC-PRA", "0", "negative");
    setAssay(6, "20260811_05",      60, "CDC-PRA", "28.57", "poly");
    setAssay(7, "20260811_06",      60, "CDC-PRA", "0", "negative");
    setAssay(8, "20260811_07",      60, "CDC-PRA", "0", "negative");
    setAssay(9, "20260811_08",      60, "CDC-PRA", "0", "negative");
}


function setAssay(index, assayName, expectedCount, assayType, pra, specificity) {
    AV_ASSAY_NAMES[index] = assayName;
    AV_ASSAY_EXPECTED[index] = expectedCount;
    AV_ASSAY_TYPE[index] = assayType;
    AV_ASSAY_PRA[index] = pra;
    AV_ASSAY_SPECIFICITY[index] = specificity;
}


function configureOutputFolders() {
    AV_PROJECT_DIR = AV_OUTPUT_PARENT + "AlloViewer_well_scoring" + File.separator;
    finalRoot = AV_PROJECT_DIR + "final" + File.separator;
    centerRoot = finalRoot + AV_CENTER + File.separator;
    AV_FINAL_DIR = centerRoot + "annotator_" + AV_SLOT + File.separator;
    draftRoot = AV_PROJECT_DIR + "drafts" + File.separator;
    draftCenter = draftRoot + AV_CENTER + File.separator;
    AV_DRAFT_DIR = draftCenter + "annotator_" + AV_SLOT + File.separator;
    AV_COMBINED_DIR = AV_PROJECT_DIR + "combined" + File.separator;

    makeDirectoryIfNeeded(AV_PROJECT_DIR);
    makeDirectoryIfNeeded(finalRoot);
    makeDirectoryIfNeeded(centerRoot);
    makeDirectoryIfNeeded(AV_FINAL_DIR);
    makeDirectoryIfNeeded(draftRoot);
    makeDirectoryIfNeeded(draftCenter);
    makeDirectoryIfNeeded(AV_DRAFT_DIR);
    makeDirectoryIfNeeded(AV_COMBINED_DIR);

    if (AV_CENTER=="Frankfurt")
        centerSlots = newArray(3, 4);
    else
        centerSlots = newArray(2, 3, 4);

    AV_OTHER_SLOT_COUNT = centerSlots.length-1;
    AV_OTHER_FINAL_DIRS = newArray(AV_OTHER_SLOT_COUNT);
    AV_OTHER_DRAFT_DIRS = newArray(AV_OTHER_SLOT_COUNT);
    otherPosition = 0;
    for (slotI=0; slotI<centerSlots.length; slotI++) {
        slotValue = centerSlots[slotI];
        if (slotValue==AV_SLOT) continue;
        AV_OTHER_FINAL_DIRS[otherPosition] = centerRoot + "annotator_" + slotValue + File.separator;
        AV_OTHER_DRAFT_DIRS[otherPosition] = draftCenter + "annotator_" + slotValue + File.separator;
        otherPosition++;
    }
}


function setCurrentAssay(index) {
    AV_CURRENT_ASSAY_INDEX = index;
    AV_CURRENT_ASSAY = AV_ASSAY_NAMES[index];
    AV_CURRENT_EXPECTED = AV_ASSAY_EXPECTED[index];
    AV_CURRENT_TYPE = AV_ASSAY_TYPE[index];
    AV_CURRENT_PRA = AV_ASSAY_PRA[index];
    AV_CURRENT_SPECIFICITY = AV_ASSAY_SPECIFICITY[index];
    AV_ASSAY_KEY = assayKey(AV_CENTER, AV_CURRENT_ASSAY);
    AV_FINAL_PATH = AV_FINAL_DIR + AV_ASSAY_KEY + ".csv";
    AV_DRAFT_PATH = AV_DRAFT_DIR + AV_ASSAY_KEY + ".wip.tsv";
}


function countCompletedAssays() {
    completedCount = 0;
    for (completeI=0; completeI<AV_ASSAY_COUNT; completeI++) {
        completePath = AV_FINAL_DIR + assayKey(AV_CENTER, AV_ASSAY_NAMES[completeI]) + ".csv";
        if (File.exists(completePath)) completedCount++;
    }
    return completedCount;
}


function findNextEligibleAssay() {
    for (eligibleI=0; eligibleI<AV_ASSAY_COUNT; eligibleI++) {
        if (AV_SESSION_SKIP[eligibleI]!=0) continue;
        if (assayIsEligible(eligibleI)) return eligibleI;
    }
    return -1;
}


function assayIsEligible(index) {
    candidateKey = assayKey(AV_CENTER, AV_ASSAY_NAMES[index]);
    candidateFinal = AV_FINAL_DIR + candidateKey + ".csv";
    if (File.exists(candidateFinal)) return 0;

    candidateDraft = AV_DRAFT_DIR + candidateKey + ".wip.tsv";
    owner = draftOwner(candidateDraft);
    if (owner!="" && toLowerCase(owner)!=toLowerCase(AV_ANNOTATOR)) return 0;

    if (sameAnnotatorUsedInOtherSlot(candidateKey)) return 0;
    return 1;
}


function chooseOneIncompleteAssay() {
    incompleteLabels = newArray(AV_ASSAY_COUNT);
    incompleteIndexes = newArray(AV_ASSAY_COUNT);
    incompleteCount = 0;

    for (incompleteI=0; incompleteI<AV_ASSAY_COUNT; incompleteI++) {
        if (!assayIsEligible(incompleteI)) continue;
        incompleteKey = assayKey(AV_CENTER, AV_ASSAY_NAMES[incompleteI]);
        incompleteDraft = AV_DRAFT_DIR + incompleteKey + ".wip.tsv";
        incompleteLabel = AV_ASSAY_NAMES[incompleteI];
        if (File.exists(incompleteDraft)) incompleteLabel = incompleteLabel + "  [resume draft]";
        incompleteLabels[incompleteCount] = incompleteLabel;
        incompleteIndexes[incompleteCount] = incompleteI;
        incompleteCount++;
    }

    if (incompleteCount==0) return -1;
    incompleteLabels = Array.trim(incompleteLabels, incompleteCount);
    incompleteIndexes = Array.trim(incompleteIndexes, incompleteCount);

    Dialog.create("Choose one incomplete assay");
    Dialog.addChoice("Assay:", incompleteLabels, incompleteLabels[0]);
    Dialog.addMessage(
        "Only per-assay final CSVs define completion.\n"+
        "The existing combined CSV is ignored and will be rebuilt after scoring."
    );
    Dialog.show();
    chosenIncomplete = Dialog.getChoice();

    for (chosenI=0; chosenI<incompleteCount; chosenI++)
        if (incompleteLabels[chosenI]==chosenIncomplete) return incompleteIndexes[chosenI];
    return -1;
}


function sameAnnotatorUsedInOtherSlot(candidateKey) {
    for (otherI=0; otherI<AV_OTHER_SLOT_COUNT; otherI++) {
        otherFinalPath = AV_OTHER_FINAL_DIRS[otherI] + candidateKey + ".csv";
        otherFinalOwner = finalOwner(otherFinalPath);
        if (otherFinalOwner!="" && toLowerCase(otherFinalOwner)==toLowerCase(AV_ANNOTATOR))
            return 1;

        otherDraftPath = AV_OTHER_DRAFT_DIRS[otherI] + candidateKey + ".wip.tsv";
        otherDraftOwner = draftOwner(otherDraftPath);
        if (otherDraftOwner!="" && toLowerCase(otherDraftOwner)==toLowerCase(AV_ANNOTATOR))
            return 1;
    }
    return 0;
}


function chooseValidatedAssayFolder() {
    while (1) {
        showMessage("Select the next assay folder",
            "Pass: " + AV_CENTER + " annotator slot " + AV_SLOT +
            "\nAssay " + (AV_CURRENT_ASSAY_INDEX+1) + " of " + AV_ASSAY_COUNT +
            "\n\nSelect the folder named exactly:\n" + AV_CURRENT_ASSAY +
            "\n\nExpected numeric TIFF images: " + AV_CURRENT_EXPECTED +
            "\nExpected first mapped well: A1"+
            "\nExpected last mapped well: " + AV_WELLS[AV_CURRENT_EXPECTED-1]);

        selectedDir = getDirectory("Select assay folder: " + AV_CURRENT_ASSAY);
        selectedDir = withTrailingSeparator(selectedDir);
        selectedName = directoryName(selectedDir);

        if (selectedName!=AV_CURRENT_ASSAY) {
            chooseAgain = getBoolean(
                "Folder name mismatch\n\nExpected: " + AV_CURRENT_ASSAY +
                "\nSelected: " + selectedName +
                "\n\nNo image has been scored.",
                "Choose again", "Skip for now");
            if (chooseAgain) continue;
            return "";
        }

        if (!scanAssayImages(selectedDir)) {
            chooseAgain = getBoolean(
                "Folder validation failed\n\n" + AV_SCAN_ERROR +
                "\n\nNo image has been scored.",
                "Choose again", "Skip for now");
            if (chooseAgain) continue;
            return "";
        }

        preflight = "Assay: " + AV_CURRENT_ASSAY +
            "\nImages: " + AV_IMAGE_COUNT +
            "\nFirst: " + AV_IMAGE_NAMES[0] + "  ->  A1"+
            "\nLast: " + AV_IMAGE_NAMES[AV_IMAGE_COUNT-1] + "  ->  " + AV_WELLS[AV_IMAGE_COUNT-1] +
            "\nNumeric gaps in this folder: " + AV_GAP_COUNT;
        if (AV_GAP_COUNT>0) preflight = preflight + "\n" + AV_GAP_TEXT;
        if (File.exists(AV_DRAFT_PATH))
            preflight = preflight + "\n\nA draft exists and will be validated against these exact filenames before resuming.";
        preflight = preflight + "\n\nUse this folder?";

        useFolder = getBoolean(preflight, "Use this folder", "Choose again");
        if (useFolder) return "" + selectedDir;
    }
}


function scanAssayImages(directory) {
    AV_SCAN_ERROR = "";
    entries = getFileList(directory);
    capacity = entries.length;
    if (capacity<1) capacity = 1;
    AV_IMAGE_NAMES = newArray(capacity);
    AV_IMAGE_PATHS = newArray(capacity);
    AV_IMAGE_NUMBERS = newArray(capacity);
    AV_IMAGE_BYTES = newArray(capacity);
    AV_IMAGE_MODIFIED = newArray(capacity);
    AV_IMAGE_COUNT = 0;
    malformed = "";

    for (scanI=0; scanI<entries.length; scanI++) {
        entry = entries[scanI];
        if (endsWith(entry, "/")) continue;
        lowerEntry = toLowerCase(entry);
        isTiff = 0;
        if (endsWith(lowerEntry, ".tif")) isTiff = 1;
        if (endsWith(lowerEntry, ".tiff")) isTiff = 1;
        if (!isTiff) continue;
        if (!startsWith(lowerEntry, "bild_")) continue;

        dotAt = lastIndexOf(lowerEntry, ".");
        numberText = substring(entry, 5, dotAt);
        if (!allDigits(numberText)) {
            if (malformed!="") malformed = malformed + ", ";
            malformed = malformed + entry;
            continue;
        }
        if (AV_IMAGE_COUNT>=AV_MAX_IMAGES) {
            AV_SCAN_ERROR = "More than " + AV_MAX_IMAGES + " Bild_*.tif files were found.";
            return 0;
        }
        imagePath = directory + entry;
        AV_IMAGE_NAMES[AV_IMAGE_COUNT] = entry;
        AV_IMAGE_PATHS[AV_IMAGE_COUNT] = imagePath;
        AV_IMAGE_NUMBERS[AV_IMAGE_COUNT] = parseInt(numberText);
        AV_IMAGE_BYTES[AV_IMAGE_COUNT] = File.length(imagePath);
        AV_IMAGE_MODIFIED[AV_IMAGE_COUNT] = File.lastModified(imagePath);
        AV_IMAGE_COUNT++;
    }

    if (malformed!="") {
        AV_SCAN_ERROR = "Malformed Bild_*.tif filename(s):\n" + malformed +
            "\n\nEvery accepted TIFF must use Bild_ followed by digits.";
        return 0;
    }
    if (AV_IMAGE_COUNT!=AV_CURRENT_EXPECTED) {
        AV_SCAN_ERROR = "Expected " + AV_CURRENT_EXPECTED + " numeric Bild_*.tif files but found " + AV_IMAGE_COUNT + "."+
            "\n\nThe macro will not guess which well is missing or extra.";
        return 0;
    }

    sortImagesNumerically();
    for (dupI=1; dupI<AV_IMAGE_COUNT; dupI++) {
        if (AV_IMAGE_NUMBERS[dupI]==AV_IMAGE_NUMBERS[dupI-1]) {
            AV_SCAN_ERROR = "Duplicate numeric image number " + AV_IMAGE_NUMBERS[dupI] +
                " in:\n" + AV_IMAGE_NAMES[dupI-1] + "\n" + AV_IMAGE_NAMES[dupI];
            return 0;
        }
    }

    AV_GAP_COUNT = 0;
    AV_GAP_TEXT = "";
    for (gapI=1; gapI<AV_IMAGE_COUNT; gapI++) {
        difference = AV_IMAGE_NUMBERS[gapI]-AV_IMAGE_NUMBERS[gapI-1];
        if (difference==1) continue;
        AV_GAP_COUNT++;
        if (AV_GAP_COUNT<=6) {
            if (AV_GAP_TEXT!="") AV_GAP_TEXT = AV_GAP_TEXT + "\n";
            AV_GAP_TEXT = AV_GAP_TEXT + AV_IMAGE_NUMBERS[gapI-1] + " -> " + AV_IMAGE_NUMBERS[gapI];
        }
    }
    if (AV_GAP_COUNT>6) AV_GAP_TEXT = AV_GAP_TEXT + "\n...and " + (AV_GAP_COUNT-6) + " more gap(s).";
    return 1;
}


function sortImagesNumerically() {
    for (sortI=1; sortI<AV_IMAGE_COUNT; sortI++) {
        sortName = AV_IMAGE_NAMES[sortI];
        sortPath = AV_IMAGE_PATHS[sortI];
        sortNumber = AV_IMAGE_NUMBERS[sortI];
        sortBytes = AV_IMAGE_BYTES[sortI];
        sortModified = AV_IMAGE_MODIFIED[sortI];
        sortJ = sortI-1;
        while (sortJ>=0) {
            if (AV_IMAGE_NUMBERS[sortJ]<=sortNumber) break;
            AV_IMAGE_NAMES[sortJ+1] = AV_IMAGE_NAMES[sortJ];
            AV_IMAGE_PATHS[sortJ+1] = AV_IMAGE_PATHS[sortJ];
            AV_IMAGE_NUMBERS[sortJ+1] = AV_IMAGE_NUMBERS[sortJ];
            AV_IMAGE_BYTES[sortJ+1] = AV_IMAGE_BYTES[sortJ];
            AV_IMAGE_MODIFIED[sortJ+1] = AV_IMAGE_MODIFIED[sortJ];
            sortJ--;
        }
        AV_IMAGE_NAMES[sortJ+1] = sortName;
        AV_IMAGE_PATHS[sortJ+1] = sortPath;
        AV_IMAGE_NUMBERS[sortJ+1] = sortNumber;
        AV_IMAGE_BYTES[sortJ+1] = sortBytes;
        AV_IMAGE_MODIFIED[sortJ+1] = sortModified;
    }
}


function resetScoreArrays() {
    AV_SCORES = newArray(AV_CURRENT_EXPECTED);
    AV_SCORE_SET = newArray(AV_CURRENT_EXPECTED);
    AV_SCORE_TIMES = newArray(AV_CURRENT_EXPECTED);
    AV_NEXT_INDEX = 0;
    AV_REFERENCE_WIDTH = 0;
    AV_REFERENCE_HEIGHT = 0;
    AV_DRAFT_ERROR = "";
}


function scoreCurrentAssay() {
    while (1) {
        if (AV_NEXT_INDEX>=AV_CURRENT_EXPECTED) {
            completeText = "All " + AV_CURRENT_EXPECTED + " wells have a score.\n\n"+
                scoreDistributionText() +
                "\n\nSave this assay as final? The final CSV cannot be overwritten by the macro.";
            finishAssay = getBoolean(completeText, "Save final assay", "Review last well");
            if (finishAssay) return "finish";
            AV_NEXT_INDEX = AV_CURRENT_EXPECTED-1;
            saveDraft();
        }

        if (!openScoreImage(AV_NEXT_INDEX)) {
            saveDraft();
            return "error";
        }

        panelAction = waitForPanelAction();
        if (panelAction=="closed") {
            saveDraft();
            closeScoreImage();
            return "pause";
        }
        if (panelAction=="pause") {
            saveDraft();
            closeScoreImage();
            return "pause";
        }
        if (panelAction=="back") {
            closeScoreImage();
            moveToPreviousWell();
            saveDraft();
            continue;
        }

        scoreValue = parseInt(panelAction);
        if (!allowedScore(scoreValue)) {
            beep();
            closeScoreImage();
            AV_OPEN_ERROR = "An internal panel-mapping error produced an invalid score: " + panelAction;
            saveDraft();
            return "error";
        }
        recordCurrentScore(scoreValue);
        saveDraft();
        closeScoreImage();
    }
}


function moveToPreviousWell() {
    if (AV_NEXT_INDEX>0) AV_NEXT_INDEX--;
}


function recordCurrentScore(scoreValue) {
    AV_SCORES[AV_NEXT_INDEX] = scoreValue;
    AV_SCORE_SET[AV_NEXT_INDEX] = 1;
    AV_SCORE_TIMES[AV_NEXT_INDEX] = timeStamp();
    AV_NEXT_INDEX++;
}


function openScoreImage(index) {
    AV_OPEN_ERROR = "";
    AV_IMAGE_ID = 0;
    AV_IMAGE_IDS_BEFORE = openImageIdCsv();
    AV_OWNED_IMAGE_IDS = "";
    beforeOpen = nImages;
    IJ.redirectErrorMessages();
    open(AV_IMAGE_PATHS[index]);
    rememberNewScoreImageIds();
    if (nImages<=beforeOpen) {
        AV_OPEN_ERROR = "Fiji could not open:\n" + AV_IMAGE_PATHS[index] +
            "\n\nThe image was not scored and the current draft was retained.";
        closeScoreImage();
        return 0;
    }

    AV_IMAGE_ID = getImageID();
    getDimensions(openWidth, openHeight, openChannels, openSlices, openFrames);
    AV_SOURCE_WIDTH = openWidth;
    AV_SOURCE_HEIGHT = openHeight;
    AV_SOURCE_CHANNELS = openChannels;

    if (AV_REFERENCE_WIDTH==0) {
        AV_REFERENCE_WIDTH = AV_SOURCE_WIDTH;
        AV_REFERENCE_HEIGHT = AV_SOURCE_HEIGHT;
    }
    if (AV_SOURCE_WIDTH!=AV_REFERENCE_WIDTH || AV_SOURCE_HEIGHT!=AV_REFERENCE_HEIGHT) {
        AV_OPEN_ERROR = "Image dimensions changed within one assay.\n\nExpected: " +
            AV_REFERENCE_WIDTH + " x " + AV_REFERENCE_HEIGHT +
            "\nFound: " + AV_SOURCE_WIDTH + " x " + AV_SOURCE_HEIGHT +
            "\nFile: " + AV_IMAGE_NAMES[index];
        closeScoreImage();
        return 0;
    }

    applyFixedDisplayRange();
    if (bitDepth()!=24) run("RGB Color");
    AV_IMAGE_ID = getImageID();
    rememberNewScoreImageIds();

    AV_PANEL_WIDTH = round(AV_SOURCE_WIDTH*0.22);
    if (AV_PANEL_WIDTH<360) AV_PANEL_WIDTH = 360;
    if (AV_PANEL_WIDTH>600) AV_PANEL_WIDTH = 600;
    AV_CANVAS_WIDTH = AV_SOURCE_WIDTH+AV_PANEL_WIDTH;
    AV_PANEL_HEADER = round(AV_SOURCE_HEIGHT*0.11);
    if (AV_PANEL_HEADER<140) AV_PANEL_HEADER = 140;
    if (AV_PANEL_HEADER>260) AV_PANEL_HEADER = 260;
    AV_BUTTON_HEIGHT = floor((AV_SOURCE_HEIGHT-AV_PANEL_HEADER)/8);
    if (AV_BUTTON_HEIGHT<1) AV_BUTTON_HEIGHT = 1;

    setBackgroundColor(18, 22, 27);
    run("Canvas Size...", "width=" + AV_CANVAS_WIDTH + " height=" + AV_SOURCE_HEIGHT + " position=Top-Left zero");
    AV_IMAGE_ID = getImageID();
    rememberNewScoreImageIds();
    if (getWidth()!=AV_CANVAS_WIDTH) {
        AV_OPEN_ERROR = "Fiji could not create the scoring panel for:\n" + AV_IMAGE_NAMES[index];
        closeScoreImage();
        return 0;
    }

    rename("Score | " + AV_CURRENT_ASSAY + " | " + AV_WELLS[index] + " | " + (index+1) + "/" + AV_CURRENT_EXPECTED);
    drawScorePanel(index);
    run("Maximize");
    wait(100);
    run("Scale to Fit");
    wait(100);
    AV_IMAGE_ID = getImageID();
    rememberNewScoreImageIds();
    setTool("hand");
    attachScoreInputListeners();
    waitUntilMouseReleased();
    return 1;
}


function applyFixedDisplayRange() {
    displayDepth = bitDepth();
    if (displayDepth==24) return;

    if (AV_SOURCE_CHANNELS>1) {
        activeChannels = "";
        for (displayC=1; displayC<=AV_SOURCE_CHANNELS; displayC++) {
            Stack.setPosition(displayC, 1, 1);
            setMinAndMax(AV_DISPLAY_MIN, AV_DISPLAY_MAX);
            activeChannels = activeChannels + "1";
        }
        Stack.setDisplayMode("composite");
        Stack.setActiveChannels(activeChannels);
        Stack.setPosition(1, 1, 1);
    } else {
        setMinAndMax(AV_DISPLAY_MIN, AV_DISPLAY_MAX);
    }
}


function drawScorePanel(index) {
    panelX = AV_SOURCE_WIDTH;
    autoUpdate(false);
    setColor("#12161b");
    fillRect(panelX, 0, AV_PANEL_WIDTH, AV_SOURCE_HEIGHT);

    setJustification("center");
    infoFont = round(AV_PANEL_HEADER*0.18);
    if (infoFont<24) infoFont = 24;
    if (infoFont>46) infoFont = 46;
    setFont("SansSerif", infoFont, "bold antialiased");
    setColor("white");
    drawString("WELL " + AV_WELLS[index], panelX+AV_PANEL_WIDTH/2, infoFont+12);
    setFont("SansSerif", round(infoFont*0.70), "antialiased");
    drawString((index+1) + " / " + AV_CURRENT_EXPECTED, panelX+AV_PANEL_WIDTH/2, infoFont*2+20);
    setFont("SansSerif", round(infoFont*0.48), "bold antialiased");
    drawString("KEYS  1  2  4  6  8  0", panelX+AV_PANEL_WIDTH/2, AV_PANEL_HEADER-48);
    setFont("SansSerif", round(infoFont*0.42), "antialiased");
    drawString(AV_IMAGE_NAMES[index], panelX+AV_PANEL_WIDTH/2, AV_PANEL_HEADER-16);

    labels = newArray("1", "2", "4", "6", "8", "0", "BACK", "PAUSE");
    colors = newArray("#174a69", "#1b5b78", "#216c82", "#277d87", "#2d8d87", "#38424c", "#5b6570", "#7b2e32");
    for (buttonI=0; buttonI<8; buttonI++) {
        buttonY = AV_PANEL_HEADER + buttonI*AV_BUTTON_HEIGHT;
        buttonH = AV_BUTTON_HEIGHT-4;
        if (buttonI==7) buttonH = AV_SOURCE_HEIGHT-buttonY;
        setColor(colors[buttonI]);
        fillRect(panelX+4, buttonY+2, AV_PANEL_WIDTH-8, buttonH-2);
        setColor("#dfe7ee");
        setLineWidth(3);
        drawRect(panelX+4, buttonY+2, AV_PANEL_WIDTH-9, buttonH-3);
        if (buttonI<6)
            buttonFont = round(AV_BUTTON_HEIGHT*0.42);
        else
            buttonFont = round(AV_BUTTON_HEIGHT*0.24);
        if (buttonFont<24) buttonFont = 24;
        if (buttonFont>96) buttonFont = 96;
        setFont("SansSerif", buttonFont, "bold antialiased");
        textY = buttonY + AV_BUTTON_HEIGHT/2 + buttonFont*0.36;
        drawString(labels[buttonI], panelX+AV_PANEL_WIDTH/2, textY);
    }
    setJustification("left");
    updateDisplay();
    autoUpdate(true);
    showStatus("Assay " + AV_CURRENT_ASSAY + " | well " + AV_WELLS[index] + " | press 1, 2, 4, 6, 8 or 0");
}


function waitForPanelAction() {
    wasDown = 0;
    pressX = -1;
    pressY = -1;
    while (1) {
        if (!isOpen(AV_IMAGE_ID)) {
            detachScoreInputListeners();
            return "closed";
        }

        inputAction = pollInputAction();
        if (inputAction=="back" && AV_NEXT_INDEX==0) {
            beep();
            showStatus("Already at the first well; there is no previous score to edit.");
        } else if (inputAction!="") {
            return inputAction;
        }

        if (!isActive(AV_IMAGE_ID)) {
            showStatus("Click the scoring image once to activate it; number keys remain available.");
            wait(10);
            continue;
        }

        getCursorLoc(cursorX, cursorY, cursorZ, cursorFlags);
        mouseDown = ((cursorFlags&16)!=0);
        if (mouseDown && !wasDown) {
            pressX = cursorX;
            pressY = cursorY;
        }
        if (!mouseDown && wasDown) {
            if (pressX>=AV_SOURCE_WIDTH && cursorX>=AV_SOURCE_WIDTH) {
                action = panelActionAt(pressY);
                if (action=="back" && AV_NEXT_INDEX==0) {
                    beep();
                    showStatus("Already at the first well; there is no previous score to edit.");
                } else if (action!="") {
                    return action;
                }
            }
        }
        wasDown = mouseDown;
        wait(10);
    }
}


function panelActionAt(y) {
    if (y<AV_PANEL_HEADER) return "";
    if (y>=AV_SOURCE_HEIGHT) return "";
    buttonIndex = floor((y-AV_PANEL_HEADER)/AV_BUTTON_HEIGHT);
    if (buttonIndex<0) return "";
    if (buttonIndex==0) return "1";
    if (buttonIndex==1) return "2";
    if (buttonIndex==2) return "4";
    if (buttonIndex==3) return "6";
    if (buttonIndex==4) return "8";
    if (buttonIndex==5) return "0";
    if (buttonIndex==6) return "back";
    if (buttonIndex>=7) return "pause";
    return "";
}


function closeScoreImage() {
    detachScoreInputListeners();

    // During a setup error, collect any image that opened before the error.
    if (AV_OWNED_IMAGE_IDS=="" && AV_IMAGE_IDS_BEFORE!="")
        rememberNewScoreImageIds();

    if (AV_OWNED_IMAGE_IDS!="") {
        closeIds = split(AV_OWNED_IMAGE_IDS, ",");
        for (closeI=closeIds.length-1; closeI>=0; closeI--) {
            closeId = parseInt(closeIds[closeI]);
            if (isOpen(closeId)) {
                selectImage(closeId);
                setOption("Changes", false);
                close();
            }
        }
    }

    // Fallback for an unexpected tracking failure: retain the previous
    // single-image cleanup, but never close a pre-existing image.
    if (AV_IMAGE_ID!=0 && isOpen(AV_IMAGE_ID) &&
        !idCsvContains(AV_IMAGE_IDS_BEFORE, AV_IMAGE_ID)) {
        selectImage(AV_IMAGE_ID);
        setOption("Changes", false);
        close();
    }

    AV_IMAGE_ID = 0;
    AV_IMAGE_IDS_BEFORE = "";
    AV_OWNED_IMAGE_IDS = "";
}


function openImageIdCsv() {
    idResult = eval("script",
        "var WM=Java.type('ij.WindowManager');"+
        "var ids=WM.getIDList();"+
        "var out='';"+
        "if(ids!=null){for(var i=0;i<ids.length;i++){if(i>0)out+=',';out+=ids[i];}}"+
        "out;");
    if (idResult=="null" || idResult=="undefined") return "";
    return "" + idResult;
}


function rememberNewScoreImageIds() {
    currentIdCsv = openImageIdCsv();
    AV_OWNED_IMAGE_IDS = "";
    if (currentIdCsv=="") return;

    currentIds = split(currentIdCsv, ",");
    for (rememberI=0; rememberI<currentIds.length; rememberI++) {
        rememberId = parseInt(currentIds[rememberI]);
        if (!idCsvContains(AV_IMAGE_IDS_BEFORE, rememberId)) {
            if (AV_OWNED_IMAGE_IDS!="") AV_OWNED_IMAGE_IDS = AV_OWNED_IMAGE_IDS + ",";
            AV_OWNED_IMAGE_IDS = AV_OWNED_IMAGE_IDS + rememberId;
        }
    }
}


function idCsvContains(csvText, imageId) {
    if (csvText=="") return 0;
    return indexOf("," + csvText + ",", "," + imageId + ",")>=0;
}


function attachScoreInputListeners() {
    AV_KEY_CAPTURE_OK = 0;
    keyScript =
        "var KFM=Java.type('java.awt.KeyboardFocusManager');"+
        "var KED=Java.type('java.awt.KeyEventDispatcher');"+
        "var KE=Java.type('java.awt.event.KeyEvent');"+
        "var MA=Java.type('java.awt.event.MouseAdapter');"+
        "var MouseExtension=Java.extend(MA);"+
        "var HashMap=Java.type('java.util.HashMap');"+
        "var WM=Java.type('ij.WindowManager');"+
        "var SYS=Java.type('java.lang.System');"+
        "var UI=Java.type('javax.swing.UIManager');"+
        "var manager=KFM.getCurrentKeyboardFocusManager();"+
        "var holder='" + AV_KEY_HOLDER + "';"+
        "var actionKey='" + AV_KEY_PROPERTY + "';"+
        "var targetId=" + AV_IMAGE_ID + ";"+
        "var sourceWidth=" + AV_SOURCE_WIDTH + ";"+
        "var sourceHeight=" + AV_SOURCE_HEIGHT + ";"+
        "var canvasWidth=" + AV_CANVAS_WIDTH + ";"+
        "var panelHeader=" + AV_PANEL_HEADER + ";"+
        "var buttonHeight=" + AV_BUTTON_HEIGHT + ";"+
        "var old=UI.get(holder);"+
        "if(old!=null){"+
            "try{"+
                "var oldKeyboard=old.get('keyboard');"+
                "var oldMouse=old.get('mouse');"+
                "var oldCanvas=old.get('canvas');"+
                "if(oldKeyboard!=null)manager.removeKeyEventDispatcher(oldKeyboard);"+
                "if(oldMouse!=null&&oldCanvas!=null)oldCanvas.removeMouseListener(oldMouse);"+
            "}catch(oldStateError){try{manager.removeKeyEventDispatcher(old);}catch(oldDispatcherError){}}"+
            "UI.getDefaults().remove(holder);"+
        "}"+
        "SYS.clearProperty(actionKey);"+
        "function scoreOf(e){"+
            "var code=e.getKeyCode();"+
            "var ch=''+e.getKeyChar();"+
            "if(code==KE.VK_1||code==KE.VK_NUMPAD1||ch=='1')return '1';"+
            "if(code==KE.VK_2||code==KE.VK_NUMPAD2||ch=='2')return '2';"+
            "if(code==KE.VK_4||code==KE.VK_NUMPAD4||ch=='4')return '4';"+
            "if(code==KE.VK_6||code==KE.VK_NUMPAD6||ch=='6')return '6';"+
            "if(code==KE.VK_8||code==KE.VK_NUMPAD8||ch=='8')return '8';"+
            "if(code==KE.VK_0||code==KE.VK_NUMPAD0||ch=='0')return '0';"+
            "return null;"+
        "}"+
        "var dispatcher=new KED(function(e){"+
            "if(WM.getImage(targetId)==null)return false;"+
            "var score=scoreOf(e);"+
            "if(score==null)return false;"+
            "e.consume();"+
            "if(e.getID()==KE.KEY_RELEASED)SYS.setProperty(actionKey,score);"+
            "return true;"+
        "});"+
        "var imp=WM.getImage(targetId);"+
        "var canvas=imp==null?null:imp.getCanvas();"+
        "var mouseListener=null;"+
        "if(canvas!=null){"+
            "mouseListener=new MouseExtension({mouseReleased:function(e){"+
                "if(WM.getImage(targetId)==null)return;"+
                "if(e.getSource()==null||!e.getSource().equals(canvas))return;"+
                "var x=canvas.offScreenX(e.getX());"+
                "var y=canvas.offScreenY(e.getY());"+
                "if(x<sourceWidth||x>=canvasWidth||y<panelHeader||y>=sourceHeight)return;"+
                "var buttonIndex=Math.floor((y-panelHeader)/buttonHeight);"+
                "var action=null;"+
                "if(buttonIndex==0)action='1';"+
                "else if(buttonIndex==1)action='2';"+
                "else if(buttonIndex==2)action='4';"+
                "else if(buttonIndex==3)action='6';"+
                "else if(buttonIndex==4)action='8';"+
                "else if(buttonIndex==5)action='0';"+
                "else if(buttonIndex==6)action='back';"+
                "else if(buttonIndex>=7)action='pause';"+
                "if(action!=null)SYS.setProperty(actionKey,action);"+
            "}});"+
            "canvas.addMouseListener(mouseListener);"+
        "}"+
        "manager.addKeyEventDispatcher(dispatcher);"+
        "var state=new HashMap();"+
        "state.put('keyboard',dispatcher);"+
        "if(mouseListener!=null)state.put('mouse',mouseListener);"+
        "if(canvas!=null)state.put('canvas',canvas);"+
        "UI.put(holder,state);"+
        "if(canvas!=null)canvas.requestFocusInWindow();"+
        "'OK';";
    keyResult = eval("script", keyScript);
    if (keyResult=="OK") {
        AV_KEY_CAPTURE_OK = 1;
        return;
    }

    if (getArgument()=="KEYSELFTEST") {
        print("KEYSELFTEST attach error: " + keyResult);
        return;
    }

    if (AV_KEY_WARNING_SHOWN==0) {
        AV_KEY_WARNING_SHOWN = 1;
        showMessage("AlloViewer keyboard fallback",
            "Fiji could not activate the temporary number-key listener.\n\n"+
            "You can continue safely by clicking the score buttons in the right-hand panel.\n\n"+
            "Technical detail:\n" + keyResult);
    }
}


function pollInputAction() {
    if (AV_KEY_CAPTURE_OK==0) return "";
    inputValue = call("java.lang.System.getProperty", AV_KEY_PROPERTY, "");
    if (inputValue=="") return "";
    previousInputValue = call("java.lang.System.clearProperty", AV_KEY_PROPERTY);
    if (inputValue=="1" || inputValue=="2" || inputValue=="4" ||
        inputValue=="6" || inputValue=="8" || inputValue=="0" ||
        inputValue=="back" || inputValue=="pause")
        return "" + inputValue;
    return "";
}


function detachScoreInputListeners() {
    keyDetachScript =
        "var KFM=Java.type('java.awt.KeyboardFocusManager');"+
        "var SYS=Java.type('java.lang.System');"+
        "var UI=Java.type('javax.swing.UIManager');"+
        "var manager=KFM.getCurrentKeyboardFocusManager();"+
        "var holder='" + AV_KEY_HOLDER + "';"+
        "var old=UI.get(holder);"+
        "if(old!=null){"+
            "try{"+
                "var oldKeyboard=old.get('keyboard');"+
                "var oldMouse=old.get('mouse');"+
                "var oldCanvas=old.get('canvas');"+
                "if(oldKeyboard!=null)manager.removeKeyEventDispatcher(oldKeyboard);"+
                "if(oldMouse!=null&&oldCanvas!=null)oldCanvas.removeMouseListener(oldMouse);"+
            "}catch(oldStateError){try{manager.removeKeyEventDispatcher(old);}catch(oldDispatcherError){}}"+
            "UI.getDefaults().remove(holder);"+
        "}"+
        "SYS.clearProperty('" + AV_KEY_PROPERTY + "');"+
        "'OK';";
    keyDetachResult = eval("script", keyDetachScript);
    AV_KEY_CAPTURE_OK = 0;
}


function waitUntilMouseReleased() {
    if (!isOpen(AV_IMAGE_ID)) return;
    getCursorLoc(waitX, waitY, waitZ, waitFlags);
    while ((waitFlags&16)!=0) {
        wait(20);
        if (!isOpen(AV_IMAGE_ID)) return;
        getCursorLoc(waitX, waitY, waitZ, waitFlags);
    }
}


function saveDraft() {
    String.resetBuffer;
    String.append("#macro_version\t" + AV_VERSION + "\n");
    String.append("#center\t" + AV_CENTER + "\n");
    String.append("#annotation_slot\t" + AV_SLOT + "\n");
    String.append("#assay\t" + AV_CURRENT_ASSAY + "\n");
    String.append("#annotator_id\t" + AV_ANNOTATOR + "\n");
    String.append("#source_folder\t" + AV_SOURCE_DIR + "\n");
    String.append("#expected_images\t" + AV_CURRENT_EXPECTED + "\n");
    String.append("#next_index\t" + AV_NEXT_INDEX + "\n");
    String.append("well_index\twell\timage_filename\timage_number\tscore\tscored_at_local\n");
    for (draftI=0; draftI<AV_NEXT_INDEX; draftI++) {
        String.append("" + (draftI+1) + "\t" + AV_WELLS[draftI] + "\t" +
            AV_IMAGE_NAMES[draftI] + "\t" + AV_IMAGE_NUMBERS[draftI] + "\t" +
            AV_SCORES[draftI] + "\t" + AV_SCORE_TIMES[draftI] + "\n");
    }
    File.saveString(String.buffer, AV_DRAFT_PATH);
}


function loadDraft() {
    if (!File.exists(AV_DRAFT_PATH)) return 0;
    draftText = replace(File.openAsString(AV_DRAFT_PATH), "\r", "");
    draftLines = split(draftText, "\n");
    storedCenter = "";
    storedSlot = "";
    storedAssay = "";
    storedOwner = "";
    storedExpected = "";
    dataStart = -1;

    for (draftLineI=0; draftLineI<draftLines.length; draftLineI++) {
        draftLine = draftLines[draftLineI];
        if (startsWith(draftLine, "#center\t")) storedCenter = substring(draftLine, 8);
        if (startsWith(draftLine, "#annotation_slot\t")) storedSlot = substring(draftLine, 17);
        if (startsWith(draftLine, "#assay\t")) storedAssay = substring(draftLine, 7);
        if (startsWith(draftLine, "#annotator_id\t")) storedOwner = substring(draftLine, 14);
        if (startsWith(draftLine, "#expected_images\t")) storedExpected = substring(draftLine, 17);
        if (startsWith(draftLine, "well_index\t")) dataStart = draftLineI+1;
    }

    if (storedCenter!=AV_CENTER || storedSlot!=(""+AV_SLOT) || storedAssay!=AV_CURRENT_ASSAY) {
        AV_DRAFT_ERROR = "Draft metadata does not match the selected task.\n\nDraft:\n" + AV_DRAFT_PATH;
        return -1;
    }
    if (toLowerCase(storedOwner)!=toLowerCase(AV_ANNOTATOR)) {
        AV_DRAFT_ERROR = "This annotation slot has a draft owned by another annotator.\n\nDraft owner: " + storedOwner +
            "\nCurrent annotator: " + AV_ANNOTATOR +
            "\n\nAsk the project lead to review the draft; do not overwrite it.";
        return -1;
    }
    if (storedExpected!=(""+AV_CURRENT_EXPECTED) || dataStart<0) {
        AV_DRAFT_ERROR = "Draft structure is incomplete or has the wrong expected image count.\n\nDraft:\n" + AV_DRAFT_PATH;
        return -1;
    }

    loadedRows = 0;
    for (draftLineI=dataStart; draftLineI<draftLines.length; draftLineI++) {
        if (trim(draftLines[draftLineI])=="") continue;
        draftCols = split(draftLines[draftLineI], "\t");
        if (draftCols.length<6) {
            AV_DRAFT_ERROR = "Draft data row " + (draftLineI+1) + " is incomplete.";
            return -1;
        }
        if (loadedRows>=AV_CURRENT_EXPECTED) {
            AV_DRAFT_ERROR = "Draft contains more rows than expected for this assay.";
            return -1;
        }
        parsedIndex = parseInt(draftCols[0]);
        parsedNumber = parseInt(draftCols[3]);
        parsedScore = parseInt(draftCols[4]);
        if (parsedIndex!=(loadedRows+1) || draftCols[1]!=AV_WELLS[loadedRows]) {
            AV_DRAFT_ERROR = "Draft well order does not match the fixed vertical-snake order at row " + (draftLineI+1) + ".";
            return -1;
        }
        if (draftCols[2]!=AV_IMAGE_NAMES[loadedRows] || parsedNumber!=AV_IMAGE_NUMBERS[loadedRows]) {
            AV_DRAFT_ERROR = "Selected folder no longer matches the draft at well " + AV_WELLS[loadedRows] +
                ".\n\nDraft file: " + draftCols[2] +
                "\nSelected folder file: " + AV_IMAGE_NAMES[loadedRows];
            return -1;
        }
        if (!allowedScore(parsedScore)) {
            AV_DRAFT_ERROR = "Draft contains invalid score " + draftCols[4] + " at well " + AV_WELLS[loadedRows] + ".";
            return -1;
        }
        AV_SCORES[loadedRows] = parsedScore;
        AV_SCORE_SET[loadedRows] = 1;
        AV_SCORE_TIMES[loadedRows] = draftCols[5];
        loadedRows++;
    }
    AV_NEXT_INDEX = loadedRows;
    return 1;
}


function saveFinalAssay() {
    if (File.exists(AV_FINAL_PATH)) return 0;
    if (AV_NEXT_INDEX!=AV_CURRENT_EXPECTED) return 0;
    for (validateI=0; validateI<AV_CURRENT_EXPECTED; validateI++)
        if (AV_SCORE_SET[validateI]==0 || !allowedScore(AV_SCORES[validateI])) return 0;

    completedAt = timeStamp();
    String.resetBuffer;
    String.append(finalHeader());
    for (finalI=0; finalI<AV_CURRENT_EXPECTED; finalI++) {
        rowText = "" + AV_ANNOTATOR + "," + AV_SLOT + "," + csvQuote(AV_CENTER) + "," +
            csvQuote(AV_CURRENT_ASSAY) + "," + csvQuote(AV_CURRENT_TYPE) + "," +
            csvQuote(AV_CURRENT_PRA) + "," + csvQuote(AV_CURRENT_SPECIFICITY) + "," +
            (finalI+1) + "," + csvQuote(AV_WELLS[finalI]) + "," +
            csvQuote(AV_IMAGE_NAMES[finalI]) + "," + AV_IMAGE_NUMBERS[finalI] + "," +
            AV_IMAGE_BYTES[finalI] + "," + AV_IMAGE_MODIFIED[finalI] + "," +
            AV_SCORES[finalI] + "," + csvQuote(AV_SCORE_TIMES[finalI]) + "," +
            csvQuote(completedAt) + "," + csvQuote(AV_SOURCE_DIR) + "," +
            csvQuote("numeric Bild_ suffix ascending") + "," + csvQuote("Vertical_snake") + "," +
            csvQuote(AV_VERSION) + "," + AV_DISPLAY_MIN + "," + AV_DISPLAY_MAX + "\n";
        String.append(rowText);
    }

    finalText = String.buffer;
    tempPath = AV_FINAL_PATH + ".tmp_" + safeComponent(AV_ANNOTATOR);
    File.saveString(finalText, tempPath);
    if (!File.exists(tempPath)) return 0;
    if (File.exists(AV_FINAL_PATH)) return 0;
    renamed = File.rename(tempPath, AV_FINAL_PATH);
    if (!renamed) return 0;
    if (!File.exists(AV_FINAL_PATH)) return 0;
    return 1;
}


function finalHeader() {
    return "annotator_id,annotation_slot,center,assay,source_assay_type,source_pra,"+
        "source_specificity,well_index,well,image_filename,image_number,image_bytes,"+
        "image_last_modified_ms,score,scored_at_local,assay_completed_at_local,"+
        "source_folder,sort_rule,well_order,macro_version,display_min,display_max\n";
}


function buildCombinedPassFile() {
    AV_LAST_COMBINED_ASSAYS = 0;
    AV_LAST_COMBINED_ROWS = 0;
    AV_LAST_COMBINED_OK = 0;
    String.resetBuffer;
    String.append(finalHeader());
    for (combineI=0; combineI<AV_ASSAY_COUNT; combineI++) {
        combinePath = AV_FINAL_DIR + assayKey(AV_CENTER, AV_ASSAY_NAMES[combineI]) + ".csv";
        if (!File.exists(combinePath)) continue;
        AV_LAST_COMBINED_ASSAYS++;
        combineText = replace(File.openAsString(combinePath), "\r", "");
        combineLines = split(combineText, "\n");
        for (combineLineI=1; combineLineI<combineLines.length; combineLineI++) {
            if (trim(combineLines[combineLineI])=="") continue;
            String.append(combineLines[combineLineI] + "\n");
            AV_LAST_COMBINED_ROWS++;
        }
    }
    expectedCombinedText = String.buffer;
    File.saveString(expectedCombinedText, combinedPassPath());
    if (!File.exists(combinedPassPath())) return 0;
    savedCombinedText = replace(File.openAsString(combinedPassPath()), "\r", "");
    if (savedCombinedText!=expectedCombinedText) return 0;
    AV_LAST_COMBINED_OK = 1;
    return 1;
}


function combinedPassPath() {
    return "" + AV_COMBINED_DIR + AV_CENTER + "_annotator_" + AV_SLOT + "_combined_long.csv";
}


function scoreDistributionText() {
    count0=0; count1=0; count2=0; count4=0; count6=0; count8=0;
    for (distributionI=0; distributionI<AV_CURRENT_EXPECTED; distributionI++) {
        value = AV_SCORES[distributionI];
        if (value==0) count0++;
        if (value==1) count1++;
        if (value==2) count2++;
        if (value==4) count4++;
        if (value==6) count6++;
        if (value==8) count8++;
    }
    return "Score counts: 1=" + count1 + ", 2=" + count2 + ", 4=" + count4 +
        ", 6=" + count6 + ", 8=" + count8 + ", 0=" + count0;
}


function nextWellText() {
    if (AV_NEXT_INDEX>=AV_CURRENT_EXPECTED) return "completion confirmation";
    return AV_WELLS[AV_NEXT_INDEX] + " (" + (AV_NEXT_INDEX+1) + "/" + AV_CURRENT_EXPECTED + ")";
}


function finalOwner(path) {
    if (!File.exists(path)) return "";
    ownerText = replace(File.openAsString(path), "\r", "");
    ownerLines = split(ownerText, "\n");
    if (ownerLines.length<2) return "";
    commaAt = indexOf(ownerLines[1], ",");
    if (commaAt<1) return "";
    return "" + trim(substring(ownerLines[1], 0, commaAt));
}


function draftOwner(path) {
    if (!File.exists(path)) return "";
    ownerText = replace(File.openAsString(path), "\r", "");
    ownerLines = split(ownerText, "\n");
    for (ownerI=0; ownerI<ownerLines.length; ownerI++) {
        if (startsWith(ownerLines[ownerI], "#annotator_id\t"))
            return "" + trim(substring(ownerLines[ownerI], 14));
    }
    return "UNKNOWN_DRAFT_OWNER";
}


function allowedScore(value) {
    if (value==0) return 1;
    if (value==1) return 1;
    if (value==2) return 1;
    if (value==4) return 1;
    if (value==6) return 1;
    if (value==8) return 1;
    return 0;
}


function allDigits(value) {
    if (lengthOf(value)==0) return 0;
    for (digitI=0; digitI<lengthOf(value); digitI++) {
        oneCharacter = substring(value, digitI, digitI+1);
        if (indexOf("0123456789", oneCharacter)<0) return 0;
    }
    return 1;
}


function validAnnotatorId(value) {
    if (lengthOf(value)==0) return 0;
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .-_";
    for (idI=0; idI<lengthOf(value); idI++) {
        oneCharacter = substring(value, idI, idI+1);
        if (indexOf(allowed, oneCharacter)<0) return 0;
    }
    return 1;
}


function directoryName(path) {
    normalized = replace(path, "\\", "/");
    while (endsWith(normalized, "/") && lengthOf(normalized)>1)
        normalized = substring(normalized, 0, lengthOf(normalized)-1);
    slashAt = lastIndexOf(normalized, "/");
    if (slashAt<0) return "" + normalized;
    return "" + substring(normalized, slashAt+1);
}


function assayKey(center, assay) {
    // Fiji/ImageJ 1.54p returns 0 for the former IJ.checksum string call.
    // All existing project finals therefore use "__0". Keep that exact,
    // deterministic convention so progress, resume and combined rebuilds
    // work on every annotator's existing files.
    return "" + safeComponent(assay) + "__0";
}


function safeComponent(value) {
    value = replace(value, " ", "_");
    value = replace(value, "/", "_");
    value = replace(value, "\\", "_");
    value = replace(value, ":", "_");
    value = replace(value, ",", "_");
    value = replace(value, "\"", "_");
    value = replace(value, "+", "plus");
    return "" + value;
}


function withTrailingSeparator(path) {
    if (path=="") return "";
    if (!endsWith(path, File.separator)) path = path + File.separator;
    return "" + path;
}


function makeDirectoryIfNeeded(path) {
    if (!File.exists(path)) File.makeDirectory(path);
    if (!File.exists(path)) exit("Could not create output folder:\n" + path);
}


function csvQuote(value) {
    quote = fromCharCode(34);
    value = "" + value;
    value = replace(value, quote, quote+quote);
    return "" + quote + value + quote;
}


function timeStamp() {
    getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
    stamp = "" + year + "-" + pad2(month+1) + "-" + pad2(dayOfMonth) + " " +
        pad2(hour) + ":" + pad2(minute) + ":" + pad2(second);
    return "" + stamp;
}


function pad2(value) {
    value = round(value);
    if (value<10) return "0"+value;
    return ""+value;
}


function runSelfTests() {
    requires("1.54i");
    setBatchMode(true);
    if (AV_WELLS.length!=60) exit("SELFTEST failed: well count");
    if (AV_WELLS[0]!="A1" || AV_WELLS[59]!="A10") exit("SELFTEST failed: well endpoints");

    configurePass("Frankfurt - annotator slot 3 (43 assay folders)");
    if (AV_CENTER!="Frankfurt" || AV_SLOT!=3 || AV_ASSAY_COUNT!=43)
        exit("SELFTEST failed: Frankfurt slot 3");
    if (AV_ASSAY_NAMES[16]!="20251021_25720338") exit("SELFTEST failed: Frankfurt order");
    if (AV_ASSAY_EXPECTED[17]!=54 || AV_ASSAY_EXPECTED[18]!=54)
        exit("SELFTEST failed: CDC-XM image counts");
    configurePass("Frankfurt - annotator slot 4 (43 assay folders)");
    if (AV_CENTER!="Frankfurt" || AV_SLOT!=4 || AV_ASSAY_COUNT!=43)
        exit("SELFTEST failed: Frankfurt slot 4");

    configurePass("Berlin - annotator slot 2 (10 assay folders)");
    if (AV_CENTER!="Berlin" || AV_SLOT!=2 || AV_ASSAY_COUNT!=10)
        exit("SELFTEST failed: Berlin slot 2");
    configurePass("Berlin - annotator slot 3 (10 assay folders)");
    if (AV_CENTER!="Berlin" || AV_SLOT!=3 || AV_ASSAY_NAMES[9]!="20260811_08")
        exit("SELFTEST failed: Berlin slot 3");
    configurePass("Berlin - annotator slot 4 (10 assay folders)");
    if (AV_CENTER!="Berlin" || AV_SLOT!=4 || AV_ASSAY_NAMES[9]!="20260811_08")
        exit("SELFTEST failed: Berlin slot 4");

    selfDirectory = directoryName("/tmp/example_assay/");
    if (selfDirectory!="example_assay")
        exit("SELFTEST failed: directory name");
    if (!allDigits("001418") || allDigits("14x18"))
        exit("SELFTEST failed: numeric suffix validation");
    if (!allowedScore(0) || !allowedScore(8) || allowedScore(3))
        exit("SELFTEST failed: score validation");
    if (!validAnnotatorId("AB_12-3") || validAnnotatorId("AB,12"))
        exit("SELFTEST failed: annotator ID validation");
    if (!idCsvContains("-1,-20,-305", -20) || idCsvContains("-1,-20,-305", -2))
        exit("SELFTEST failed: image ID tracking");

    AV_CURRENT_EXPECTED = 2;
    resetScoreArrays();
    recordCurrentScore(2);
    moveToPreviousWell();
    recordCurrentScore(8);
    if (AV_NEXT_INDEX!=1 || AV_SCORES[0]!=8 || AV_SCORE_SET[0]!=1)
        exit("SELFTEST failed: BACK/rescore overwrite behavior");

    macroDir = File.getDirectory(getInfo("macro.filepath"));
    AV_CURRENT_EXPECTED = 1;
    if (!scanAssayImages(macroDir + "upload" + File.separator))
        exit("SELFTEST failed: example TIFF scan: " + AV_SCAN_ERROR);
    if (AV_IMAGE_COUNT!=1 || AV_IMAGE_NAMES[0]!="Bild_1418.tif" || AV_IMAGE_NUMBERS[0]!=1418)
        exit("SELFTEST failed: example TIFF mapping");

    testDir = getDirectory("temp") + "AlloViewer_well_score_selftest_" + round(getTime()) + File.separator;
    File.makeDirectory(testDir);
    if (!File.exists(testDir)) exit("SELFTEST failed: temp directory");
    AV_CENTER = "SelfTest";
    AV_SLOT = 3;
    AV_ANNOTATOR = "SELFTEST";
    AV_CURRENT_ASSAY = "SELFTEST_ASSAY";
    AV_CURRENT_EXPECTED = 1;
    AV_CURRENT_TYPE = "CDC-PRA";
    AV_CURRENT_PRA = "NA";
    AV_CURRENT_SPECIFICITY = "NA";
    AV_SOURCE_DIR = macroDir + "upload" + File.separator;
    AV_DRAFT_PATH = testDir + "draft.wip.tsv";
    AV_FINAL_PATH = testDir + "final.csv";
    resetScoreArrays();
    AV_SCORES[0] = 4;
    AV_SCORE_SET[0] = 1;
    AV_SCORE_TIMES[0] = "2026-10-23 12:00:00";
    AV_NEXT_INDEX = 1;
    saveDraft();
    if (!File.exists(AV_DRAFT_PATH)) exit("SELFTEST failed: draft save");
    resetScoreArrays();
    if (loadDraft()<0 || AV_NEXT_INDEX!=1 || AV_SCORES[0]!=4)
        exit("SELFTEST failed: draft resume");
    if (!saveFinalAssay() || !File.exists(AV_FINAL_PATH))
        exit("SELFTEST failed: atomic final save");
    if (saveFinalAssay()) exit("SELFTEST failed: final overwrite protection");
    savedFinal = File.openAsString(AV_FINAL_PATH);
    if (indexOf(savedFinal, "annotator_id,annotation_slot")!=0 || indexOf(savedFinal, ",4,")<0)
        exit("SELFTEST failed: final CSV content");
    deletedDraft = File.delete(AV_DRAFT_PATH);
    deletedFinal = File.delete(AV_FINAL_PATH);
    deletedDirectory = File.delete(testDir);

    print("SELFTEST PASS: Frankfurt/Berlin slot selection through annotator 4, image ID tracking, manifest, numeric TIFF scan, score validation, BACK/rescore overwrite, draft resume, atomic final save and overwrite protection OK.");
}


function runKeyboardBridgeSelfTest() {
    requires("1.54i");
    setBatchMode(true);
    newImage("AlloViewer keyboard self-test", "8-bit black", 16, 16, 1);
    AV_IMAGE_ID = getImageID();
    attachScoreInputListeners();
    if (AV_KEY_CAPTURE_OK==0) {
        print("KEYSELFTEST failed: dispatcher did not attach");
        return;
    }

    dispatchTestScript =
        "var UI=Java.type('javax.swing.UIManager');"+
        "var KE=Java.type('java.awt.event.KeyEvent');"+
        "var Canvas=Java.type('java.awt.Canvas');"+
        "var state=UI.get('" + AV_KEY_HOLDER + "');"+
        "if(state==null)throw 'missing input-listener state';"+
        "var d=state.get('keyboard');"+
        "if(d==null)throw 'missing keyboard dispatcher';"+
        "var e=new KE(new Canvas(),KE.KEY_RELEASED,java.lang.System.currentTimeMillis(),0,KE.VK_4,'4');"+
        "d.dispatchKeyEvent(e);"+
        "'OK';";
    dispatchResult = eval("script", dispatchTestScript);
    if (dispatchResult!="OK") {
        print("KEYSELFTEST failed: " + dispatchResult);
        return;
    }
    capturedTestScore = pollInputAction();
    if (capturedTestScore!="4") {
        print("KEYSELFTEST failed: score 4 was not captured");
        return;
    }

    detachScoreInputListeners();
    detachedResult = eval("script",
        "var UI=Java.type('javax.swing.UIManager');"+
        "UI.get('" + AV_KEY_HOLDER + "')==null?'OK':'FAILED';");
    if (detachedResult!="OK") {
        print("KEYSELFTEST failed: dispatcher was not removed");
        return;
    }
    setOption("Changes", false);
    close();
    print("KEYSELFTEST PASS: number-key dispatcher attached, captured score 4 on release and detached cleanly.");
}
