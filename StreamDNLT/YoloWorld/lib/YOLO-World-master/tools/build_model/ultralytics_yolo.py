from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO("yolov8s-world.pt")  # or choose yolov8m/l-world.pt

# Define custom classes
model.set_classes(["whiteairplanelandingontheground"])

# Execute prediction for specified categories on an image
results = model.predict("/home/share/hhd/dataset/lgt/uvltrack_work/data/lasot/airplane/airplane-1/img/00000001.jpg")

# Show results
results[0].save("lib/YOLO-World-master/tools/build_model/output/anno.jpg")