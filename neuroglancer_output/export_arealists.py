from ij import IJ, ImagePlus, ImageStack
from ij.measure import Calibration
from ij.process import ByteProcessor
from ij.gui import ShapeRoi

from java.io import File
from java.awt.geom import Area, AffineTransform

from ini.trakem2.Project import getProjects
from ini.trakem2.display import LayerSet, Layer, AreaTree

# Set destination directory and create it if it doesn't exist
destdir = "C:/Users/steve/Desktop/testareatree/"
File(destdir).mkdirs()

# Get current project
project = getProjects().get(0)
if project is None:
    IJ.log("No project is open!")
else:
    # Get root and LayerSet
    root = project.getRootProjectThing()
    layerSet = project.getRootLayerSet()

    # Get all layers in the project
    layers = layerSet.getLayers()
    IJ.log("Project has {} layers".format(layers.size()))

    # Counter for exported trees
    exportCount = 0

    # Create an image stack from an AreaTree
    def createStackFromAreaTree(areaTree, layers):
        # Fixed reasonable dimensions
        width = 512
        height = 512
        
        # Create the image stack
        stack = ImageStack(width, height)
        
        # Process each layer directly
        layersWithContent = 0
        
        for layer in layers:
            # Try to get area for this layer
            try:
                area = areaTree.getAreaAt(layer)
                if area is not None and not area.isEmpty():
                    # Create a mask for this layer
                    bp = ByteProcessor(width, height)
                    bp.setValue(255)  # White
                
                    # Scale the area to fit our image dimensions
                    bounds = area.getBounds()
                    
                    # Create a scaled version that fits in our image
                    scaleX = (width * 0.9) / bounds.width if bounds.width > 0 else 1.0
                    scaleY = (height * 0.9) / bounds.height if bounds.height > 0 else 1.0
                    scale = min(scaleX, scaleY)
                    
                    # Center in the image
                    offsetX = (width - int(bounds.width * scale)) / 2
                    offsetY = (height - int(bounds.height * scale)) / 2
                    
                    # Create transformation
                    at = AffineTransform()
                    at.translate(offsetX - bounds.x * scale, offsetY - bounds.y * scale)
                    at.scale(scale, scale)
                    
                    # Transform the area
                    transformedArea = area.createTransformedArea(at)
                    
                    # Convert the area to a proper ROI
                    shapeRoi = ShapeRoi(transformedArea)
                    
                    # Draw into the mask
                    bp.fill(shapeRoi)
                    
                    # Add to stack
                    stack.addSlice("Layer {}".format(layer.getZ()), bp)
                    layersWithContent += 1
            except Exception as e:
                IJ.log("Error processing layer {}: {}".format(layer.getZ(), str(e)))
        
        IJ.log("AreaTree has {} layers with content".format(layersWithContent))
        
        if layersWithContent > 0:
            imp = ImagePlus("AreaTree Mask", stack)
            cal = layerSet.getCalibrationCopy()
            imp.setCalibration(cal)
            return imp
        else:
            return None

    # Walk through the project tree and find all AreaTrees
    def findAndExportAreaTrees(thing, path):
        global exportCount
        
        if thing is None:
            return
        
        # Current path including this thing
        currentPath = path + "_" + thing.getTitle()
        
        # Check if this thing is an AreaTree
        obj = thing.getObject()
        if isinstance(obj, AreaTree):
            areaTree = obj
            
            try:
                IJ.log("Processing AreaTree: {}".format(currentPath))
                
                # Create image stack
                imp = createStackFromAreaTree(areaTree, layers)
                
                # Save the image if it contains data
                if imp is not None and imp.getStackSize() > 0:
                    filename = destdir + currentPath + ".tif"
                    IJ.log("Saving to: {}".format(filename))
                    IJ.save(imp, filename)
                    exportCount += 1
                else:
                    IJ.log("No content in AreaTree: {}".format(currentPath))
            except Exception as e:
                IJ.log("Error processing {}: {}".format(currentPath, str(e)))
        
        # Process children
        children = thing.getChildren()
        if children is not None:
            for child in children:
                findAndExportAreaTrees(child, currentPath)

    # Start processing from the root
    IJ.log("Starting export to: {}".format(destdir))
    exportCount = 0
    findAndExportAreaTrees(root, "")
    IJ.log("Export complete! Exported {} AreaTrees".format(exportCount))