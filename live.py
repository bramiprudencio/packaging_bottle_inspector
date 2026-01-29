import cv2
import numpy as np
from ultralytics import RTDETR
import supervision as sv

def main():
    # --- CONFIGURACIÓN ---
    print("Cargando modelo RT-DETR...")
    model = RTDETR('./rtdetr-x_1.pt')

    # --- SELECCIÓN DE CÁMARA ---
    camera_index = 2  # Cambia esto según la cámara que quieras probar (4, 2, 0)
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ ERROR: No se pudo abrir la cámara {camera_index}.")
        return

    # Ajustar resolución (Alta resolución)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Configuramos los anotadores
    # Box: Solo dibuja el cuadro, sin texto (para no ensuciar la imagen)
    box_annotator = sv.BoxAnnotator(thickness=2)
    # Label: Si quieres texto en la imagen, déjalo, si no, puedes quitarlo.
    # Aquí lo dejo pero más pequeño.
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)

    print(f"✅ Cámara {camera_index} iniciada. Presiona 'q' para salir.")

    nombre_ventana = f"Test Camara {camera_index} - Con Leyenda"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 1600, 900) 
    
    # Ancho de la barra lateral en pixeles
    ANCHO_BARRA = 700 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame.")
            break

        # --- 1. INFERENCIA ---
        results = model(frame, verbose=False, conf=0.15)[0] # Conf bajo para ver todo
        detections = sv.Detections.from_ultralytics(results)

        # --- 2. PREPARAR ETIQUETAS PARA VISUALIZACIÓN ---
        labels_img = [] # Etiquetas para pintar sobre la botella
        lista_errores = [] # Lista de datos para la leyenda lateral

        for class_id, confidence in zip(detections.class_id, detections.confidence):
            nombre_clase = model.names[class_id]
            texto_full = f"{nombre_clase}: {confidence:.1%}"
            
            labels_img.append(f"{nombre_clase}") # En la imagen solo el nombre corto
            lista_errores.append((nombre_clase, confidence))

        # --- 3. DIBUJAR SOBRE EL FRAME (Boxes) ---
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels_img)

        # --- 4. CREAR EL LIENZO (CANVAS) CON BARRA LATERAL ---
        h, w, _ = frame.shape
        # Creamos una imagen negra más ancha (Ancho original + Barra)
        canvas = np.zeros((h, w + ANCHO_BARRA, 3), dtype=np.uint8)
        
        # Pegamos el frame de la cámara a la izquierda
        canvas[0:h, 0:w] = frame
        
        # --- 5. DIBUJAR LA LEYENDA A LA DERECHA ---
        # Coordenada X donde empieza la barra lateral (un poco de margen)
        x_text = w + 20 
        y_text = 50

        # Título de la leyenda
        cv2.putText(canvas, "RESULTADOS DETECTADOS:", (x_text, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        y_text += 60

        if len(lista_errores) == 0:
            cv2.putText(canvas, "--- SIN DETECCIONES ---", (x_text, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        else:
            # Ordenamos por confianza (el error más seguro arriba)
            lista_errores.sort(key=lambda x: x[1], reverse=True)

            for nombre, conf in lista_errores:
                # Color del texto: Verde si es OK, Rojo/Naranja si es Error
                color = (0, 0, 255) # Rojo por defecto (BGR)
                if "ok" in nombre.lower():
                    color = (0, 255, 0) # Verde
                
                texto_linea = f"{nombre.upper()}"
                texto_score = f"{conf:.1%}"

                # Escribimos nombre del error
                cv2.putText(canvas, texto_linea, (x_text, y_text), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Escribimos el porcentaje al lado
                cv2.putText(canvas, texto_score, (x_text + 460, y_text), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1)
                
                y_text += 50 # Salto de línea

        # --- MOSTRAR EL CANVAS COMPLETO ---
        cv2.imshow(nombre_ventana, canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()