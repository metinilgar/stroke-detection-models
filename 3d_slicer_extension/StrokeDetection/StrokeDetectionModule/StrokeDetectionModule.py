import logging
import os
from typing import Annotated, Optional, Dict, List

import vtk
import numpy as np
import qt

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLSegmentationNode

from StrokeModels.classification import StrokeClassificationModel
from StrokeModels.segmentation import StrokeSegmentationModel


#
# StrokeDetectionModule
#


class StrokeDetectionModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)        
        self.parent.title = _("Stroke Detection Module")
        self.parent.categories = ["Stroke Analysis"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name (Your Organization)"]
        self.parent.helpText = _("""
        This module provides tools for stroke detection and segmentation on brain CT images using deep learning models.
        The module allows for comparing automatic detections with ground truth masks and displays slice-by-slice results.
        """)
        self.parent.acknowledgementText = _("""
        This module was developed as part of a project for stroke detection and analysis.
        """)

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # StrokeDetectionModule1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="StrokeDetectionModule",
        sampleName="StrokeDetectionModule1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "StrokeDetectionModule1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="StrokeDetectionModule1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="StrokeDetectionModule1",
    )


#
# StrokeDetectionModuleParameterNode
#


@parameterNodeWrapper
class StrokeDetectionModuleParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to analyze for stroke.
    maskVolume - The mask volume to verify stroke areas.
    """
    inputVolume: vtkMRMLScalarVolumeNode
    maskVolume: vtkMRMLScalarVolumeNode


#
# StrokeDetectionModuleWidget
#


class StrokeDetectionModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        self._updatingGUIFromParameterNode = False

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/StrokeDetectionModule.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = StrokeDetectionModuleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)        # These connections ensure that whenever user changes some settings, the changes are saved
        # to the parameter node (so that when the user returns, their settings are restored)
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onInputVolumeChanged)
        self.ui.maskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onMaskVolumeChanged)

        # Button connections
        self.ui.classificationButton.connect("clicked(bool)", self.onClassificationButton)
        self.ui.segmentationButton.connect("clicked(bool)", self.onSegmentationButton)
        self.ui.show3DCheckBox.connect("toggled(bool)", self.onShow3DToggled)
        
        # Dataset connections
        self.ui.browseButton.connect("clicked(bool)", self.onBrowseButton)
        self.ui.classifyDatasetButton.connect("clicked(bool)", self.onClassifyDatasetButton)
        self.ui.segmentDatasetButton.connect("clicked(bool)", self.onSegmentDatasetButton)

        # Initialize instance variables for segmentation nodes
        self.predicted_segmentation_node = None
        self.true_segmentation_node = None

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode: Optional[StrokeDetectionModuleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()    
    def _checkCanApply(self, caller=None, event=None) -> None:
        """
        Enable or disable the classification and segmentation buttons based on input volume selection
        """
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.maskVolume:
            self.ui.classificationButton.enabled = True
        else:
            self.ui.classificationButton.enabled = False
            self.ui.segmentationButton.enabled = False
            
        # Update results label
        if not self._parameterNode or not self._parameterNode.inputVolume:
            self.ui.resultsLabel.setText("No results available - select an input volume")
        elif not self._parameterNode.maskVolume:
            self.ui.resultsLabel.setText("No results available - select a mask volume")    
            
    def onInputVolumeChanged(self, node) -> None:
        """
        Called when the input volume is changed.
        Disable the segmentation button as it requires classification to be run first.
        """
        self.ui.segmentationButton.enabled = False
        self.ui.resultsLabel.setText("No results available")
        
    def onMaskVolumeChanged(self, node) -> None:
        """
        Called when the mask volume is changed.
        Disable the segmentation button as it requires classification to be run first.
        """
        self.ui.segmentationButton.enabled = False
        self.ui.resultsLabel.setText("No results available")
        
    def onClassificationButton(self) -> None:
        """
        Run the stroke classification when the Classification button is clicked.
        """
        with slicer.util.tryWithErrorDisplay("Failed to run stroke classification.", waitCursor=True):
            inputVolume = self._parameterNode.inputVolume
            maskVolume = self._parameterNode.maskVolume
            if not inputVolume or not maskVolume:
                return
            logging.info("Starting stroke classification...")
            self.ui.resultsLabel.setText("Running classification...")
            slicer.app.processEvents()  # Update UI
            
            # Run classification
            slice_results, max_stroke_slice, avg_probability = self.logic.classifyVolumeWithMask(inputVolume, maskVolume)
            
            # Calculate accuracy statistics
            total_slices = len(slice_results)
            correct_predictions = 0
            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0
            
            for result in slice_results.values():
                pred_class = 1 if result["predicted_class"] == 1 else 0
                actual_class = 0 if result["mask_empty"] else 1
                
                if pred_class == actual_class:
                    correct_predictions += 1
                    if pred_class == 1:
                        true_positives += 1
                    else:
                        true_negatives += 1
                else:
                    if pred_class == 1 and actual_class == 0:
                        false_positives += 1
                    elif pred_class == 0 and actual_class == 1:
                        false_negatives += 1
            
            accuracy = (correct_predictions / total_slices * 100) if total_slices > 0 else 0
            
            # Format results in a clean table with improved styling
            results_text = "<style>"
            results_text += "table { border-collapse: collapse; width: 100%; margin-bottom: 15px; }"
            results_text += "th, td { padding: 8px; text-align: center; }"
            results_text += "th { background-color: #4CAF50; color: white; }"
            results_text += "tr:nth-child(even) { background-color: #f2f2f2; }"
            results_text += ".summary { margin-top: 15px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; border: 1px solid #d0e3e8; }"
            results_text += ".correct { background-color: #64de81; }"
            results_text += ".incorrect { background-color: #fa7582; }"
            results_text += "</style>"
            
            # Add the main results table
            results_text += "<table border='1'>"
            results_text += "<tr><th>Slice</th><th>Prediction</th><th>Ground Truth</th><th>Probability</th></tr>"
            
            # Data rows
            for slice_idx, result in sorted(slice_results.items()):
                pred_class = "1" if result["predicted_class"] == 1 else "0"
                actual_class = "1" if not result["mask_empty"] else "0"
                prob = result["stroke_probability"] * 100
                
                # Determine if prediction is correct
                row_class = "correct" if pred_class == actual_class else "incorrect"
                
                # Add row to table with class for styling
                results_text += f"<tr class='{row_class}'><td>Slice {slice_idx}</td><td>{pred_class}</td><td>{actual_class}</td><td>{prob:.2f}%</td></tr>"
            
            results_text += "</table>"
            
            # Add accuracy statistics
            results_text += "<div class='summary'>"
            results_text += f"<p><b>ACCURACY:</b> {accuracy:.2f}% ({correct_predictions}/{total_slices} slices correctly classified)</p>"
            results_text += "<p><b>DETAILED STATISTICS:</b></p>"
            results_text += "<table border='1' style='width: 50%; margin: 0 auto;'>"
            results_text += "<tr><th colspan='2' rowspan='2'></th><th colspan='2'>Actual</th></tr>"
            results_text += "<tr><th>Stroke (1)</th><th>No Stroke (0)</th></tr>"
            results_text += f"<tr><th rowspan='2'>Predicted</th><th>Stroke (1)</th><td>{true_positives}</td><td>{false_positives}</td></tr>"
            results_text += f"<tr><th>No Stroke (0)</th><td>{false_negatives}</td><td>{true_negatives}</td></tr>"
            results_text += "</table>"
            
            # Add summary information
            if max_stroke_slice >= 0:
                # Jump to the slice with maximum stroke probability
                self.logic.jumpToSlice(inputVolume, max_stroke_slice)
                
                # Add summary to results
                results_text += f"<p><b>SUMMARY:</b> Stroke Presence ({avg_probability:.2f}%)</p>"
                results_text += f"<p><b>Maximum stroke probability on slice {max_stroke_slice}</b></p>"
                # Enable segmentation button
                self.ui.segmentationButton.enabled = True
            else:
                results_text += "<p><b>SUMMARY:</b> No Stroke Detected</p>"
                self.ui.segmentationButton.enabled = False
            
            results_text += "</div>"
            
            # Display results in the UI
            self.ui.resultsLabel.setText(results_text)
            logging.info("Stroke classification completed")    
            
    def onSegmentationButton(self) -> None:
        """
        Run the stroke segmentation when the Segmentation button is clicked.
        """
        with slicer.util.tryWithErrorDisplay("Failed to run stroke segmentation.", waitCursor=True):
            inputVolume = self._parameterNode.inputVolume
            maskVolume = self._parameterNode.maskVolume
            if not inputVolume or not maskVolume:
                return
            logging.info("Starting stroke segmentation...")
            current_text = self.ui.resultsLabel.toPlainText()
            self.ui.resultsLabel.setText(current_text + "\nRunning segmentation...")
            slicer.app.processEvents()  # Update UI
            
            # Run segmentation
            self.predicted_segmentation_node, self.true_segmentation_node = self.logic.segmentVolumeWithMask(inputVolume, maskVolume)
            
            # Get the evaluation metrics from the logic
            dice_score = self.logic.last_dice_score
            tn, fp, fn, tp = self.logic.last_confusion_matrix
            
            # Create HTML table for results
            results_html = """
            <style>
                .metrics-table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0;
                }
                .metrics-table th, .metrics-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }
                .metrics-table th {
                    background-color: #4CAF50;
                    color: white;
                }
                .metrics-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .dice-score {
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #4CAF50;
                    margin: 10px 0;
                }
            </style>
            """
            
            # Add Dice score
            results_html += f'<div class="dice-score">Dice Skoru: {dice_score:.4f}</div>'
            
            # Add confusion matrix table
            results_html += """
            <table class="metrics-table">
                <tr>
                    <th colspan="2" rowspan="2"></th>
                    <th colspan="2">Gerçek Değer</th>
                </tr>
                <tr>
                    <th>Pozitif (1)</th>
                    <th>Negatif (0)</th>
                </tr>
                <tr>
                    <th rowspan="2">Tahmin</th>
                    <th>Pozitif (1)</th>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
                <tr>
                    <th>Negatif (0)</th>
                    <td>{}</td>
                    <td>{}</td>
                </tr>
            </table>
            """.format(tp, fp, fn, tn)
            
            # Update the results label with the new HTML content
            self.ui.resultsLabel.setText(results_html)
            
            # Enable 3D checkbox
            self.ui.show3DCheckBox.setEnabled(True)
            
            logging.info("Stroke segmentation completed")
            
    def onShow3DToggled(self, checked: bool) -> None:
        """
        Toggle 3D visualization of segmentations when the Show 3D checkbox is toggled
        
        Parameters:
        -----------
        checked : bool
            Whether the checkbox is checked or not
        """
        if not self.predicted_segmentation_node or not self.true_segmentation_node:
            return
            
        # Set 3D visibility based on checkbox state
        if self.predicted_segmentation_node.GetDisplayNode():
            self.predicted_segmentation_node.GetDisplayNode().SetVisibility3D(checked)
        if self.true_segmentation_node.GetDisplayNode():
            self.true_segmentation_node.GetDisplayNode().SetVisibility3D(checked)
        
        if checked:
            # Set segmentation display properties for better 3D visualization
            # Predicted segmentation - Red
            predicted_display_node = self.predicted_segmentation_node.GetDisplayNode()
            if predicted_display_node:
                predicted_display_node.SetOpacity3D(0.7)
            
            # True segmentation - Green
            true_display_node = self.true_segmentation_node.GetDisplayNode()
            if true_display_node:
                true_display_node.SetOpacity3D(0.7)
            
            # Switch to a layout that includes 3D view
            layoutManager = slicer.app.layoutManager()
            layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
            
            # Reset 3D camera to show the segmentations
            threeDWidget = layoutManager.threeDWidget(0)
            if threeDWidget:
                threeDView = threeDWidget.threeDView()
                threeDView.resetFocalPoint()
            
            logging.info("3D visualization enabled")
        else:
            # Switch back to conventional view but keep the layout
            logging.info("3D visualization disabled")

    def onBrowseButton(self) -> None:
        """Handle browse button click to select dataset directory"""
        dataset_dir = qt.QFileDialog.getExistingDirectory(
            self.parent,
            "Veri Seti Klasörünü Seç",
            "",
            qt.QFileDialog.ShowDirsOnly
        )
        if dataset_dir:
            self.ui.datasetPathLineEdit.setText(dataset_dir)

    def onClassifyDatasetButton(self) -> None:
        """Handle classify dataset button click"""
        dataset_dir = self.ui.datasetPathLineEdit.text
        if not dataset_dir:
            slicer.util.errorDisplay("Lütfen bir veri seti klasörü seçin.")
            return

        # İlerleme çubuğunu sıfırla
        self.ui.progressBar.setValue(0)
        self.ui.progressLabel.setText("İşlem başlatılıyor...")
        self.ui.classifyDatasetButton.setEnabled(False)
        slicer.app.processEvents()

        with slicer.util.tryWithErrorDisplay("Veri seti sınıflandırma başarısız oldu.", waitCursor=True):
            self.logic.classifyDataset(dataset_dir, self.updateProgress, self.updateResults)
            
        # İşlem bittiğinde butonu tekrar aktif et
        self.ui.classifyDatasetButton.setEnabled(True)
        self.ui.progressLabel.setText("Hazır")

    def updateProgress(self, current: int, total: int, message: str) -> None:
        """
        İlerleme çubuğunu güncelle
        
        Parameters:
        -----------
        current : int
            Mevcut işlem sayısı
        total : int
            Toplam işlem sayısı
        message : str
            Gösterilecek mesaj
        """
        progress = int((current / total) * 100)
        self.ui.progressBar.setValue(progress)
        self.ui.progressLabel.setText(message)
        slicer.app.processEvents()

    def updateResults(self, results_html: str) -> None:
        """
        Sonuçları resultsLabel'a yaz
        
        Parameters:
        -----------
        results_html : str
            HTML formatında sonuçlar
        """
        self.ui.resultsLabel.setHtml(results_html)
        slicer.app.processEvents()

    def onSegmentDatasetButton(self) -> None:
        """Handle segment dataset button click"""
        dataset_dir = self.ui.datasetPathLineEdit.text
        if not dataset_dir:
            slicer.util.errorDisplay("Lütfen bir veri seti klasörü seçin.")
            return

        # İlerleme çubuğunu sıfırla
        self.ui.progressBar.setValue(0)
        self.ui.progressLabel.setText("İşlem başlatılıyor...")
        self.ui.segmentDatasetButton.setEnabled(False)
        slicer.app.processEvents()

        with slicer.util.tryWithErrorDisplay("Veri seti segmentasyonu başarısız oldu.", waitCursor=True):
            self.logic.segmentDataset(dataset_dir, self.updateProgress, self.updateResults)
            
        # İşlem bittiğinde butonu tekrar aktif et
        self.ui.segmentDatasetButton.setEnabled(True)
        self.ui.progressLabel.setText("Hazır")


#
# StrokeDetectionModuleLogic
#


class StrokeDetectionModuleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        
        # Initialize model paths
        self.model_folder = os.path.join(os.path.dirname(__file__), "Resources", "Models")
        self.classification_model_path = os.path.join(self.model_folder, "classification_model.pth")
        self.segmentation_model_path = os.path.join(self.model_folder, "segmentation_model.pth")
        
        # Store slice classification results for use in segmentation
        self.slice_classification_results = {}
        self.stroke_slices = []
        
        # Store last evaluation metrics
        self.last_dice_score = 0.0
        self.last_confusion_matrix = (0, 0, 0, 0)  # (tn, fp, fn, tp)

    def getParameterNode(self):
        return StrokeDetectionModuleParameterNode(super().getParameterNode())

    def classifyVolumeWithMask(self, inputVolume: vtkMRMLScalarVolumeNode, maskVolume: vtkMRMLScalarVolumeNode) -> tuple:
        """
        Classify each slice of the input volume for stroke presence and compare with the mask
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            Input volume to classify
        maskVolume : vtkMRMLScalarVolumeNode
            Mask volume for verification
            
        Returns:
        --------
        tuple
            (slice_results, max_stroke_slice, avg_probability)
            slice_results: Dictionary of slice predictions with mask status
            max_stroke_slice: Index of the slice with highest stroke probability
            avg_probability: Average probability of stroke across all positive slices
        """
        if not inputVolume or not maskVolume:
            return {}, -1, 0.0
            
        # Load the classification model
        try:
            classifier = StrokeClassificationModel(self.classification_model_path)
        except Exception as e:
            logging.error(f"Failed to load classification model: {e}")
            return {}, -1, 0.0
            
        # Get the volume arrays
        volume_array = slicer.util.arrayFromVolume(inputVolume)
        mask_array = slicer.util.arrayFromVolume(maskVolume)
        
        # Get volume dimensions
        dims = volume_array.shape
        
        # Results storage
        slice_results = {}
        self.slice_classification_results = {}
        self.stroke_slices = []
        
        max_stroke_prob = -1
        max_stroke_slice = -1
        total_stroke_prob = 0
        stroke_count = 0
        
        # Iterate through axial slices
        for i in range(dims[0]):
            # Get the slice image and mask
            slice_image = volume_array[i, :, :]
            mask_slice = mask_array[i, :, :]
            
            # Check if mask slice is empty (all zeros/black)
            mask_empty = np.all(mask_slice <= 0)
            
            # Classify the slice
            result = classifier.predict(slice_image)
            
            # Add mask information to the result
            result["mask_empty"] = mask_empty
            result["actual_class"] = 0 if mask_empty else 1  # 0 = no stroke, 1 = stroke
            
            # Store the results
            slice_results[i] = result
            self.slice_classification_results[i] = result
            
            # Check if stroke is detected
            if result["predicted_class"] == 1:  # Stroke class
                self.stroke_slices.append(i)
                stroke_count += 1
                stroke_prob = result["stroke_probability"] * 100  # Convert to percentage
                total_stroke_prob += stroke_prob
                
                # Track the slice with highest stroke probability
                if stroke_prob > max_stroke_prob:
                    max_stroke_prob = stroke_prob
                    max_stroke_slice = i
        
        # Calculate average probability if any stroke detected
        avg_probability = (total_stroke_prob / stroke_count) if stroke_count > 0 else 0
        
        return slice_results, max_stroke_slice, avg_probability

    def jumpToSlice(self, inputVolume: vtkMRMLScalarVolumeNode, slice_idx: int) -> None:
        """
        Jump to the specified axial slice of the input volume
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            Input volume
        slice_idx : int
            Index of the axial slice to display
        """
        # Center the 3D view on the volume
        slicer.util.resetSliceViews()
        
        # Get the axial slice view
        axial_view = slicer.app.layoutManager().sliceWidget("Red")
        if axial_view:
            # Calculate the RAS position of the slice
            ijk_to_ras = vtk.vtkMatrix4x4()
            inputVolume.GetIJKToRASMatrix(ijk_to_ras)
            
            # Get the origin of the volume
            origin = inputVolume.GetOrigin()
            
            # Calculate the IJK position of the slice, considering the origin
            slice_position = [0, 0, slice_idx, 1]  # Only Z (axial) position is adjusted
            ras_position = ijk_to_ras.MultiplyPoint(slice_position)
            
            # Jump to the selected slice position
            axial_view.sliceLogic().SetSliceOffset(ras_position[2])  # Z position in RAS space
            
            # Show the volume in all viewers
            slicer.util.setSliceViewerLayers(background=inputVolume)

    def segmentVolumeWithMask(self, inputVolume: vtkMRMLScalarVolumeNode, maskVolume: vtkMRMLScalarVolumeNode) -> tuple:
        """
        Segment stroke regions in the input volume using the segmentation model and verify with the mask
        
        Parameters:
        -----------
        inputVolume : vtkMRMLScalarVolumeNode
            Input volume to segment
        maskVolume : vtkMRMLScalarVolumeNode
            Mask volume for verification
            
        Returns:
        --------
        tuple
            (predicted_segmentation_node, true_segmentation_node) - the segmentation nodes for predicted and true (mask) stroke regions
        """
        import numpy as np
        
        if not inputVolume or not self.stroke_slices:
            logging.error("No input volume or no stroke slices identified")
            return None, None
            
        try:
            # Load the segmentation model
            segmenter = StrokeSegmentationModel(self.segmentation_model_path)
        except Exception as e:
            logging.error(f"Failed to load segmentation model: {e}")
            return None, None
        
        # Get the volume array and mask array
        volume_array = slicer.util.arrayFromVolume(inputVolume)
        mask_array = slicer.util.arrayFromVolume(maskVolume)
        
        # Create binary label maps with the same dimensions as the input volume
        dims = volume_array.shape
        predicted_label_map = np.zeros(dims, dtype=np.uint8)
        true_label_map = np.zeros(dims, dtype=np.uint8)
        
        # Process only the slices that were classified as having a stroke
        for slice_idx in self.stroke_slices:
            # Get the slice image
            slice_image = volume_array[slice_idx, :, :]
            
            # Get segmentation mask for the slice (predicted)
            pred_mask = segmenter.predict_mask(slice_image)
            
            # Get the ground truth mask from the mask volume
            true_mask_slice = mask_array[slice_idx, :, :]
            
            # Check if the true mask is empty (all zeros/black)
            mask_empty = np.all(true_mask_slice <= 0)
            
            # Resize the predicted mask to match the original slice dimensions
            if pred_mask.shape != slice_image.shape:
                from scipy.ndimage import zoom
                zoom_factors = (slice_image.shape[0] / pred_mask.shape[0], slice_image.shape[1] / pred_mask.shape[1])
                pred_mask = zoom(pred_mask, zoom_factors, order=0)  # order=0 for nearest neighbor
            
            # Add the masks to the label maps
            predicted_label_map[slice_idx, :, :] = (pred_mask > 0.5).astype(np.uint8)
            if not mask_empty:
                true_label_map[slice_idx, :, :] = (true_mask_slice > 0).astype(np.uint8)

        # Calculate confusion matrix manually
        pred_flat = predicted_label_map.flatten()
        true_flat = true_label_map.flatten()
        
        # Initialize confusion matrix elements
        tp = np.sum((pred_flat == 1) & (true_flat == 1))
        tn = np.sum((pred_flat == 0) & (true_flat == 0))
        fp = np.sum((pred_flat == 1) & (true_flat == 0))
        fn = np.sum((pred_flat == 0) & (true_flat == 1))

        # Calculate Dice score
        dice_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Create segmentation nodes
        predicted_segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        predicted_segmentation_node.SetName("Predicted_Stroke_Segmentation")
        
        true_segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        true_segmentation_node.SetName("True_Stroke_Segmentation")
        
        # Create temporary label map volumes
        pred_temp_label_map = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        pred_temp_label_map.SetName("temp_predicted_stroke_label_map")
        
        true_temp_label_map = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        true_temp_label_map.SetName("temp_true_stroke_label_map")
        
        # Copy volume attributes from the input volume
        pred_temp_label_map.CopyOrientation(inputVolume)
        true_temp_label_map.CopyOrientation(inputVolume)
        
        # Update the label maps with our segmentation results
        slicer.util.updateVolumeFromArray(pred_temp_label_map, predicted_label_map)
        slicer.util.updateVolumeFromArray(true_temp_label_map, true_label_map)
        
        # Import the label maps to the segmentation nodes
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(pred_temp_label_map, predicted_segmentation_node)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(true_temp_label_map, true_segmentation_node)
        
        # Rename and color the segments
        pred_segment_ids = vtk.vtkStringArray()
        predicted_segmentation_node.GetSegmentation().GetSegmentIDs(pred_segment_ids)
        if pred_segment_ids.GetNumberOfValues() > 0:
            pred_segment_id = pred_segment_ids.GetValue(0)
            segment = predicted_segmentation_node.GetSegmentation().GetSegment(pred_segment_id)
            segment.SetName("Predicted Stroke")
            segment.SetColor(1, 0, 0)  # Red for predicted
        
        true_segment_ids = vtk.vtkStringArray()
        true_segmentation_node.GetSegmentation().GetSegmentIDs(true_segment_ids)
        if true_segment_ids.GetNumberOfValues() > 0:
            true_segment_id = true_segment_ids.GetValue(0)
            segment = true_segmentation_node.GetSegmentation().GetSegment(true_segment_id)
            segment.SetName("True Stroke")
            segment.SetColor(0, 1, 0)  # Green for ground truth
        
        # Remove the temporary label map nodes
        slicer.mrmlScene.RemoveNode(pred_temp_label_map)
        slicer.mrmlScene.RemoveNode(true_temp_label_map)
        
        # Display only the predicted segmentation in 2D by default
        slicer.util.setSliceViewerLayers(label=predicted_segmentation_node.GetID())
        
        # Hide segmentations in 3D view by default
        predicted_segmentation_node.GetDisplayNode().SetVisibility3D(False)
        true_segmentation_node.GetDisplayNode().SetVisibility3D(False)

        self.last_dice_score = dice_score
        self.last_confusion_matrix = (tn, fp, fn, tp)
        
        return predicted_segmentation_node, true_segmentation_node

    def classifyDataset(self, dataset_dir: str, progress_callback=None, results_callback=None) -> None:
        """
        Tüm veri setini sınıflandır ve genel confusion matrix hesapla
        
        Parameters:
        -----------
        dataset_dir : str
            Veri seti klasörünün yolu
        progress_callback : callable, optional
            İlerleme durumunu güncellemek için callback fonksiyonu
        results_callback : callable, optional
            Sonuçları güncellemek için callback fonksiyonu
        """
        import os
        import glob
        import numpy as np
        
        # Model yükle
        try:
            classifier = StrokeClassificationModel(self.classification_model_path)
        except Exception as e:
            logging.error(f"Model yüklenemedi: {e}")
            return
            
        # Image ve mask klasörlerini kontrol et
        image_dir = os.path.join(dataset_dir, "image")
        mask_dir = os.path.join(dataset_dir, "mask")
        
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            logging.error("Image veya mask klasörü bulunamadı")
            return
            
        # Tüm .nii.gz dosyalarını bul
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        
        if not image_files or not mask_files:
            logging.error("Veri setinde .nii.gz dosyası bulunamadı")
            return
            
        # Confusion matrix için sayaçlar
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        
        # Toplam işlem sayısını hesapla (hasta sayısı)
        total_patients = len(image_files)
        current_patient = 0
        
        # Her hasta için
        for img_path, mask_path in zip(image_files, mask_files):
            current_patient += 1
            patient_name = os.path.basename(img_path)
            
            if progress_callback:
                progress_callback(current_patient, total_patients, f"Hasta işleniyor: {patient_name}")
            
            # Dosya isimlerini kontrol et
            if os.path.basename(img_path) != os.path.basename(mask_path):
                logging.warning(f"Eşleşmeyen dosya çifti: {img_path} - {mask_path}")
                continue
                
            # Görüntüleri yükle
            try:
                img_volume = slicer.util.loadVolume(img_path)
                mask_volume = slicer.util.loadVolume(mask_path)
            except Exception as e:
                logging.error(f"Dosya yüklenemedi {img_path}: {e}")
                continue
                
            # Görüntü dizilerini al
            img_array = slicer.util.arrayFromVolume(img_volume)
            mask_array = slicer.util.arrayFromVolume(mask_volume)
            
            # Her slice için
            for slice_idx in range(img_array.shape[0]):
                # Slice ve mask'i al
                slice_image = img_array[slice_idx, :, :]
                mask_slice = mask_array[slice_idx, :, :]
                
                # Mask boş mu kontrol et
                mask_empty = np.all(mask_slice <= 0)
                
                # Sınıflandır
                result = classifier.predict(slice_image)
                pred_class = result["predicted_class"]
                actual_class = 0 if mask_empty else 1
                
                # Confusion matrix'i güncelle
                if pred_class == 1 and actual_class == 1:
                    total_tp += 1
                elif pred_class == 0 and actual_class == 0:
                    total_tn += 1
                elif pred_class == 1 and actual_class == 0:
                    total_fp += 1
                elif pred_class == 0 and actual_class == 1:
                    total_fn += 1
            
            # Geçici node'ları temizle
            slicer.mrmlScene.RemoveNode(img_volume)
            slicer.mrmlScene.RemoveNode(mask_volume)
            
        if progress_callback:
            progress_callback(total_patients, total_patients, "Sonuçlar hesaplanıyor...")
            
        # Sonuçları hesapla
        total_slices = total_tp + total_tn + total_fp + total_fn
        accuracy = (total_tp + total_tn) / total_slices if total_slices > 0 else 0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Sonuçları göster
        results_html = """
        <style>
            .metrics-table {
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }
            .metrics-table th, .metrics-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            .metrics-table th {
                background-color: #4CAF50;
                color: white;
            }
            .metrics-table tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            .metrics {
                font-size: 1.2em;
                font-weight: bold;
                color: #4CAF50;
                margin: 10px 0;
            }
        </style>
        """
        
        # Metrikleri ekle
        results_html += f'<div class="metrics">'
        results_html += f'<p>Toplam Hasta Sayısı: {total_patients}</p>'
        results_html += f'<p>Toplam Slice Sayısı: {total_slices}</p>'
        results_html += f'<p>Doğruluk: {accuracy:.4f}</p>'
        results_html += f'<p>Kesinlik: {precision:.4f}</p>'
        results_html += f'<p>Duyarlılık: {recall:.4f}</p>'
        results_html += f'<p>F1 Skoru: {f1_score:.4f}</p>'
        results_html += '</div>'
        
        # Confusion matrix tablosunu ekle
        results_html += """
        <table class="metrics-table">
            <tr>
                <th colspan="2" rowspan="2"></th>
                <th colspan="2">Gerçek Değer</th>
            </tr>
            <tr>
                <th>Pozitif (1)</th>
                <th>Negatif (0)</th>
            </tr>
            <tr>
                <th rowspan="2">Tahmin</th>
                <th>Pozitif (1)</th>
                <td>{}</td>
                <td>{}</td>
            </tr>
            <tr>
                <th>Negatif (0)</th>
                <td>{}</td>
                <td>{}</td>
            </tr>
        </table>
        """.format(total_tp, total_fp, total_fn, total_tn)
        
        # Sonuçları göster
        slicer.util.showStatusMessage("Veri seti sınıflandırma tamamlandı")
        
        # Sonuçları callback ile güncelle
        if results_callback:
            results_callback(results_html)

    def segmentDataset(self, dataset_dir: str, progress_callback=None, results_callback=None) -> None:
        """
        Tüm veri setini segmente et ve ortalama dice skorunu hesapla
        
        Parameters:
        -----------
        dataset_dir : str
            Veri seti klasörünün yolu
        progress_callback : callable, optional
            İlerleme durumunu güncellemek için callback fonksiyonu
        results_callback : callable, optional
            Sonuçları güncellemek için callback fonksiyonu
        """
        import os
        import glob
        import numpy as np
        
        # Model yükle
        try:
            segmenter = StrokeSegmentationModel(self.segmentation_model_path)
        except Exception as e:
            logging.error(f"Model yüklenemedi: {e}")
            return
            
        # Image ve mask klasörlerini kontrol et
        image_dir = os.path.join(dataset_dir, "image")
        mask_dir = os.path.join(dataset_dir, "mask")
        
        if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
            logging.error("Image veya mask klasörü bulunamadı")
            return
            
        # Tüm .nii.gz dosyalarını bul
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        
        if not image_files or not mask_files:
            logging.error("Veri setinde .nii.gz dosyası bulunamadı")
            return
            
        # Toplam işlem sayısını hesapla (hasta sayısı)
        total_patients = len(image_files)
        current_patient = 0
        
        # Her hasta için dice skorlarını sakla
        patient_dice_scores = []
        
        # Her hasta için
        for img_path, mask_path in zip(image_files, mask_files):
            current_patient += 1
            patient_name = os.path.basename(img_path)
            
            if progress_callback:
                progress_callback(current_patient, total_patients, f"Hasta işleniyor: {patient_name}")
            
            # Dosya isimlerini kontrol et
            if os.path.basename(img_path) != os.path.basename(mask_path):
                logging.warning(f"Eşleşmeyen dosya çifti: {img_path} - {mask_path}")
                continue
                
            # Görüntüleri yükle
            try:
                img_volume = slicer.util.loadVolume(img_path)
                mask_volume = slicer.util.loadVolume(mask_path)
            except Exception as e:
                logging.error(f"Dosya yüklenemedi {img_path}: {e}")
                continue
                
            # Görüntü dizilerini al
            img_array = slicer.util.arrayFromVolume(img_volume)
            mask_array = slicer.util.arrayFromVolume(mask_volume)
            
            # Her slice için
            slice_dice_scores = []
            for slice_idx in range(img_array.shape[0]):
                # Slice ve mask'i al
                slice_image = img_array[slice_idx, :, :]
                mask_slice = mask_array[slice_idx, :, :]
                
                # Mask boş mu kontrol et
                mask_empty = np.all(mask_slice <= 0)
                
                if not mask_empty:
                    # Segmentasyon yap
                    pred_mask = segmenter.predict_mask(slice_image)
                    
                    # Resize the predicted mask to match the original slice dimensions
                    if pred_mask.shape != slice_image.shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (slice_image.shape[0] / pred_mask.shape[0], slice_image.shape[1] / pred_mask.shape[1])
                        pred_mask = zoom(pred_mask, zoom_factors, order=0)  # order=0 for nearest neighbor
                    
                    # Dice skorunu hesapla
                    pred_flat = (pred_mask > 0.5).astype(np.uint8).flatten()
                    true_flat = (mask_slice > 0).astype(np.uint8).flatten()
                    
                    tp = np.sum((pred_flat == 1) & (true_flat == 1))
                    fp = np.sum((pred_flat == 1) & (true_flat == 0))
                    fn = np.sum((pred_flat == 0) & (true_flat == 1))
                    
                    dice_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
                    slice_dice_scores.append(dice_score)
            
            # Hasta için ortalama dice skorunu hesapla
            if slice_dice_scores:
                patient_dice_scores.append(np.mean(slice_dice_scores))
            
            # Geçici node'ları temizle
            slicer.mrmlScene.RemoveNode(img_volume)
            slicer.mrmlScene.RemoveNode(mask_volume)
            
        if progress_callback:
            progress_callback(total_patients, total_patients, "Sonuçlar hesaplanıyor...")
            
        # Tüm hastalar için ortalama dice skorunu hesapla
        mean_dice_score = np.mean(patient_dice_scores) if patient_dice_scores else 0
        
        # Sonuçları göster
        results_html = """
        <style>
            .metrics {
                font-size: 1.2em;
                font-weight: bold;
                color: #4CAF50;
                margin: 10px 0;
            }
            .patient-scores {
                margin-top: 20px;
                max-height: 300px;
                overflow-y: auto;
            }
            .patient-scores table {
                width: 100%;
                border-collapse: collapse;
            }
            .patient-scores th, .patient-scores td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .patient-scores th {
                background-color: #4CAF50;
                color: white;
            }
            .patient-scores tr:nth-child(even) {
                background-color: #f2f2f2;
            }
        </style>
        """
        
        # Genel metrikleri ekle
        results_html += f'<div class="metrics">'
        results_html += f'<p>Toplam Hasta Sayısı: {total_patients}</p>'
        results_html += f'<p>Ortalama Dice Skoru: {mean_dice_score:.4f}</p>'
        results_html += '</div>'
        
        # Hasta bazlı sonuçları ekle
        results_html += '<div class="patient-scores">'
        results_html += '<table>'
        results_html += '<tr><th>Hasta</th><th>Dice Skoru</th></tr>'
        
        for patient_name, dice_score in zip([os.path.basename(f) for f in image_files], patient_dice_scores):
            results_html += f'<tr><td>{patient_name}</td><td>{dice_score:.4f}</td></tr>'
            
        results_html += '</table>'
        results_html += '</div>'
        
        # Sonuçları göster
        slicer.util.showStatusMessage("Veri seti segmentasyonu tamamlandı")
        
        # Sonuçları callback ile güncelle
        if results_callback:
            results_callback(results_html)


#
# StrokeDetectionModuleTest
#


class StrokeDetectionModuleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_StrokeDetectionModule1()    
        
    def test_StrokeDetectionModule1(self):
        """Test the basic functionality of the module."""
        self.delayDisplay("Starting the test")

        # Get/create input data
        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample("StrokeDetectionModule1")
        self.delayDisplay("Loaded test data set")
        
        # Create a test mask volume (copy of the input for testing)
        maskVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "TestMask")
        slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, inputVolume, "TestMask")

        # Test the module logic
        logic = StrokeDetectionModuleLogic()
        
        # Test classification with mask
        slice_results, max_stroke_slice, avg_probability = logic.classifyVolumeWithMask(inputVolume, maskVolume)
        self.delayDisplay(f"Classification completed, max stroke slice: {max_stroke_slice}, avg probability: {avg_probability}")
        
        if max_stroke_slice >= 0:
            # Test segmentation
            predicted_seg, true_seg = logic.segmentVolumeWithMask(inputVolume, maskVolume)
            self.assertIsNotNone(predicted_seg)
            self.assertIsNotNone(true_seg)
            self.delayDisplay("Segmentation completed")

        self.delayDisplay("Test passed")