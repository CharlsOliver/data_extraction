import os
import cv2
import pytesseract

# Esta clase convierte las imagenes etiquetadas de formato YOLO a formato IOB

# Función para leer las clases desde el archivo classes.txt
def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split('\n')
    return classes

# Función para leer las etiquetas YOLO
def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    return [list(map(float, line.strip().split())) for line in labels]

# Función para realizar OCR en la imagen y extraer el texto dentro de una bounding box
def extract_text(image, bbox):
    h, w, _ = image.shape
    x_center, y_center, bw, bh = bbox
    x1 = int((x_center - bw / 2) * w)
    y1 = int((y_center - bh / 2) * h)
    x2 = int((x_center + bw / 2) * w)
    y2 = int((y_center + bh / 2) * h)
    
    cropped_image = image[y1:y2, x1:x2]
    text = pytesseract.image_to_string(cropped_image, config='--psm 6')
    return text.strip()

# Función para convertir etiquetas YOLO a texto etiquetado
def convert_yolo_to_text(image_file, label_file, classes):
    image = cv2.imread(image_file)
    labels = load_labels(label_file)
    
    labeled_text = []
    for label in labels:
        class_id, x_center, y_center, bw, bh = label
        class_name = classes[int(class_id)]
        text = extract_text(image, (x_center, y_center, bw, bh))
        
        if text:
            words = text.split()
            for i, word in enumerate(words):
                tag = f"B-{class_name}" if i == 0 else f"I-{class_name}"
                labeled_text.append(f"{word} {tag}")
    
    return "\n".join(labeled_text)

# Directorios
images_dir = 'data/images'
labels_dir = 'data/labels'
classes_file = 'data/classes.txt'
output_dir = 'output'

# Asegurarse de que el directorio de salida exista
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Cargar clases
classes = load_classes(classes_file)

# Procesar cada imagen y sus etiquetas correspondientes
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        image_file = os.path.join(images_dir, label_file.replace('.txt', '.jpg'))
        label_file_path = os.path.join(labels_dir, label_file)
        
        labeled_text = convert_yolo_to_text(image_file, label_file_path, classes)
        
        output_file_path = os.path.join(output_dir, label_file.replace('.txt', '.ann'))
        
        # Asegurarse de que el subdirectorio de salida exista
        output_subdir = os.path.dirname(output_file_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        with open(output_file_path, 'w') as f:
            f.write(labeled_text)

        print(f"Procesado {image_file}")
