import cv2
import time
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont
from PIL import Image, ImageTk
from ultralytics import RTDETR
import supervision as sv
from opcua import Client, ua
import numpy as np

# --- 1. CONFIGURACIÓN MODULAR ---
ENABLE_CUELLO = False
ENABLE_CUERPO = True
LIMITE_LOTE = 24  

# --- 2. CONFIGURACIÓN DE VISUALIZACIÓN ---
PREVIEW_WIDTH = 530
PREVIEW_HEIGHT = 298

# --- 3. CONFIGURACIÓN DE CÁMARAS ---
CAMERAS_CONFIG = [
    { "id": 4, "desc": "CAM_DER", "foco": 228 },
    { "id": 2, "desc": "CAM_CEN", "foco": 45 },
    { "id": 0, "desc": "CAM_IZQ", "foco": 228 }
]

MODEL_PATH = './rtdetr-x_2.pt' 
OUTPUT_FOLDER = "registro_inspecciones"
OPC_URL = "opc.tcp://172.16.40.150:49340"

# ==========================================
#      CONFIGURACIÓN DE SENSIBILIDAD
# ==========================================
DEFAULT_CONFIDENCE = 0.4 

CONFIDENCE_MAP_1 = {
    'cuerpo_alreves': 0.9,
    'cuerpo_arrugada': 0.35,
    'cuerpo_ausente': 0.5,
    'cuerpo_doblada': 0.15,
    'cuerpo_falla_adherencia': 0.15,
    'cuerpo_invertida': 0.9,
    'cuerpo_ok': 0.3,
    'cuerpo_multiple': 0.8,
    'cuerpo_rasgada': 0.9,
    'cuerpo_sucia': 0.4,
    'cuerpo_torcida': 0.9
}

OPC_TAGS_MAP = {
    'cuerpo_alreves':           "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.alreves",
    'cuerpo_arrugada':          "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.arrugada",
    'cuerpo_ausente':           "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.ausente",
    'cuerpo_doblada':           "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.doblada",
    'cuerpo_arrugada_doblada':  "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.arrugada_doblada",
    'cuerpo_falla_adherencia':  "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.falla_adherencia",
    'cuerpo_invertida':         "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.invertida",
    'cuerpo_multiple':          "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.multiple",
    'cuerpo_rasgada':           "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.rasgada",
    'cuerpo_sucia':             "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.sucia",
    'cuerpo_torcida':           "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.torcida",
    'TOTAL_OK':                 "ns=2;s=SODA_SPLAN.CALIDAD.ETI_CUERPO.ok",
    'STATUS':                   "ns=2;s=SODA_SPLAN.CALIDAD.status"
}

# --- 4. DEFINICIÓN DE ERRORES ---
ERRORES_CUELLO = ['cuello_alreves', 'cuello_arrugada', 'cuello_doblada', 'cuello_falla_adherencia', 'cuello_multiple', 'cuello_rasgada']
ERRORES_CUERPO = ['cuerpo_alreves', 'cuerpo_arrugada', 'cuerpo_arrugada_doblada', 'cuerpo_doblada', 'cuerpo_falla_adherencia', 'cuerpo_invertida','cuerpo_multiple', 'cuerpo_rasgada', 'cuerpo_sucia']

ACTIVE_ERRORS = []
if ENABLE_CUELLO: ACTIVE_ERRORS.extend(ERRORES_CUELLO + ['cuello_ausente'])
if ENABLE_CUERPO: ACTIVE_ERRORS.extend(ERRORES_CUERPO + ['cuerpo_ausente'])

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------------------------------------------------
# CLASE OPC UA
# ---------------------------------------------------------
class OPCClient:
    def __init__(self, url):
        self.url = url
        self.client = Client(url)
        self.connected = False
        self.client.timeout = 2 

    def conectar(self):
        try:
            self.client.connect()
            self.connected = True
            return True, "Conectado"
        except Exception as e:
            self.connected = False
            return False, str(e)

    def actualizar_tags(self, contadores):
        if not self.connected: return False, "No conectado"
        errores_escritura = 0
        for key_interna, node_id in OPC_TAGS_MAP.items():
            try:
                valor = contadores.get(key_interna, 0)
                node = self.client.get_node(node_id)
                variant_type = node.get_data_type_as_variant_type()
                val_convertido = valor
                if variant_type in [ua.VariantType.Float, ua.VariantType.Double]: val_convertido = float(valor)
                elif variant_type in [ua.VariantType.Int16, ua.VariantType.Int32, ua.VariantType.Int64, ua.VariantType.UInt16, ua.VariantType.UInt32, ua.VariantType.UInt64]: val_convertido = int(valor)
                elif variant_type == ua.VariantType.Boolean: val_convertido = bool(valor)
                mi_variant = ua.Variant(val_convertido, variant_type)
                node.set_value(ua.DataValue(mi_variant))
            except Exception as e:
                errores_escritura += 1
        return (True, f"Enviado con {errores_escritura} errores") if errores_escritura > 0 else (True, "Tags Actualizados")

    def desconectar(self):
        try: self.client.disconnect()
        except: pass
    
    def escribir_tag_individual(self, key_interna, valor):
        if not self.connected: return False, "No conectado"
        try:
            node_id = OPC_TAGS_MAP.get(key_interna)
            if not node_id: return False, "Tag no encontrado"
            node = self.client.get_node(node_id)
            variant_type = node.get_data_type_as_variant_type()
            val_convertido = valor
            if variant_type in [ua.VariantType.Float, ua.VariantType.Double]: val_convertido = float(valor)
            elif variant_type in [ua.VariantType.Int16, ua.VariantType.Int32, ua.VariantType.Int64, ua.VariantType.UInt16, ua.VariantType.UInt32, ua.VariantType.UInt64, ua.VariantType.Byte]: val_convertido = int(valor)
            mi_variant = ua.Variant(val_convertido, variant_type)
            node.set_value(ua.DataValue(mi_variant))
            return True, "OK"
        except Exception as e:
            return False, str(e)

# ---------------------------------------------------------
# CLASE CÁMARA
# ---------------------------------------------------------
class CameraStream:
    def __init__(self, config):
        self.src = config['id']
        self.foco_target = config['foco']
        self.desc = config.get('desc', f"CAM_{self.src}")
        self.stream = None
        self.stopped = False
        self.grabbed = False
        self.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def start(self):    
        t = threading.Thread(target=self.update, args=())
        t.daemon = True 
        t.start()
        return self

    def update(self):
        print(f"[{self.desc}] Conectando en puerto {self.src}...")
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        resoluciones_a_probar = [(2304, 1536), (1920, 1080), (1280, 720)]
        resolucion_fijada = False
        for (w, h) in resoluciones_a_probar:
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            real_w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            real_h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (real_w == w and real_h == h) or (real_w > 1920):
                resolucion_fijada = True
                break
        
        self.stream.read(); time.sleep(0.5)
        self.aplicar_foco(); time.sleep(0.5)
        self.aplicar_foco(); time.sleep(0.5)
        self.aplicar_foco()

        if not self.stream.isOpened():
            self.stopped = True
            return

        while True:
            if self.stopped:
                self.stream.release()
                return
            grabbed, frame = self.stream.read()
            if grabbed:
                self.grabbed = True
                self.frame = frame
            else:
                self.grabbed = False
                time.sleep(0.01)

    def aplicar_foco(self):
        try:
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
            self.stream.set(cv2.CAP_PROP_FOCUS, self.foco_target)
        except: pass
    def read(self): return self.grabbed, self.frame
    def release(self): self.stopped = True

# ---------------------------------------------------------
# APP PRINCIPAL
# ---------------------------------------------------------
class QualityApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x900")
        try: self.window.attributes('-zoomed', True)
        except: self.window.state('normal')

        self.stats_counters = {k: 0 for k in ACTIVE_ERRORS}
        self.stats_counters['TOTAL_OK'] = 0
        self.stats_counters['TOTAL_NOK'] = 0
        self.contador_lote = 0 
        
        # --- BUFFER PARA FOTOS ---
        # Ahora guardamos un diccionario: { 'frames': [...], 'row_id': 'I001', 'time': '12:00:00' }
        self.buffer_lote = [] 

        try: self.model = RTDETR(MODEL_PATH)
        except: 
            print("⚠️ Usando modelo standard.")
            self.model = RTDETR('rtdetr-l.pt')
        
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.opc = OPCClient(OPC_URL)
        
        # --- GUI CAMARAS ---
        self.frame_cams = tk.Frame(window, bg="#2c3e50")
        self.frame_cams.pack(fill="x", ipady=5)
        
        num_cams = len(CAMERAS_CONFIG)
        for i in range(num_cams): self.frame_cams.grid_columnconfigure(i, weight=1)
        
        self.lbl_cams = []
        for i, conf in enumerate(CAMERAS_CONFIG):
            lbl_text = f"{conf['desc']} (ID:{conf['id']})"
            lbl = tk.Label(self.frame_cams, text=lbl_text, bg="black", fg="white", font=("Arial", 12, "bold"))
            lbl.grid(row=0, column=i, padx=5, pady=5)
            self.lbl_cams.append(lbl)

        # --- CONTROLES ---
        self.frame_control = tk.Frame(window, bg="#ecf0f1")
        self.frame_control.pack(fill="x", pady=5)

        info_text = "CONFIGURACIÓN ACTIVA: "
        info_text += "[✅ CUELLO] " if ENABLE_CUELLO else "[❌ CUELLO] "
        info_text += "[✅ CUERPO]" if ENABLE_CUERPO else "[❌ CUERPO]"
        tk.Label(self.frame_control, text=info_text, bg="#ecf0f1", fg="#7f8c8d", font=("Arial", 11)).pack(pady=2)

        self.frame_buttons = tk.Frame(self.frame_control, bg="#ecf0f1")
        self.frame_buttons.pack(pady=5)

        self.btn_capturar = tk.Button(self.frame_buttons, 
                                      text=f"CAPTURAR BOTELLA (0/{LIMITE_LOTE})",
                                      command=self.capturar_botella, 
                                      bg="#d35400", fg="white",
                                      font=("Arial", 16, "bold"), height=2, width=30)
        self.btn_capturar.pack(side="left", padx=10)

        self.btn_procesar = tk.Button(self.frame_buttons, 
                                      text="PROCESAR LOTE COMPLETO",
                                      command=lambda: self.procesar_lote_completo(CONFIDENCE_MAP_1), 
                                      bg="#2980b9", fg="white",
                                      font=("Arial", 16, "bold"), height=2, width=30)
        self.btn_procesar.pack(side="left", padx=10)

        self.btn_enviar_opc = tk.Button(self.frame_control, text="PUBLICAR RESULTADOS A ATHENA", 
                                      command=self.enviar_opc_manual, 
                                      bg="#27ae60", fg="white",
                                      font=("Arial", 14, "bold"), height=2, width=30)
        self.btn_enviar_opc.pack(pady=10)

        # --- TABLA ---
        self.frame_table = tk.Frame(window)
        self.frame_table.pack(fill="both", expand=True, padx=10, pady=10)
        
        mis_columnas = ["HORA", "RESULTADO"] + ACTIVE_ERRORS + ["OPC"]
        style = ttk.Style()
        style.configure("Treeview.Heading", font=('Arial', 9, 'bold'))
        style.configure("Treeview", rowheight=30, font=('Arial', 10))
        
        self.tree = ttk.Treeview(self.frame_table, columns=mis_columnas, show='headings', height=10)
        vsb = ttk.Scrollbar(self.frame_table, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.frame_table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscroll=vsb.set, xscroll=hsb.set)
        vsb.pack(side="right", fill="y"); hsb.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)
        
        for col in mis_columnas:
            text_header = col.upper()
            width = 120 
            if col == "HORA": width = 100
            elif col == "RESULTADO": text_header = "ESTADO"; width = 80
            elif col == "OPC": width = 60
            else: text_header = col.replace("cuerpo_", "").replace("cuello_", "").replace("_", " ").upper(); width = 160 
            self.tree.heading(col, text=text_header)
            self.tree.column(col, width=width, anchor="center", minwidth=80)

        vals_iniciales = ["TOTALES", "OK:0 / NOK:0"] + ["0"]*len(ACTIVE_ERRORS) + ["-"]
        self.tree.insert("", 0, iid="total_row", values=vals_iniciales, tags=("total_style",))
        self.tree.tag_configure("total_style", background="#2c3e50", foreground="white", font=('Arial', 10, 'bold')) 
        self.tree.tag_configure("nok", background="#e74c3c", foreground="white")
        self.tree.tag_configure("ok", background="#2ecc71", foreground="black")
        self.tree.tag_configure("warning", background="#f39c12", foreground="black")
        self.tree.tag_configure("pendiente", background="#bdc3c7", foreground="#7f8c8d") # Estilo gris para pendientes

        self.frame_footer = tk.Frame(window, bg="#95a5a6", height=40)
        self.frame_footer.pack(fill="x", side="bottom")
        self.lbl_opc_status = tk.Label(self.frame_footer, text="OPC UA: Desconectado", bg="#95a5a6", font=("Arial", 10))
        self.lbl_opc_status.pack(side="left", padx=20, pady=5)
        self.btn_connect_opc = tk.Button(self.frame_footer, text="Conectar OPC", command=self.conectar_opc_thread, font=("Arial", 9))
        self.btn_connect_opc.pack(side="right", padx=20, pady=5)

        self.camera_streams = []
        self.current_frames_raw = [None] * len(CAMERAS_CONFIG)
        self.window.after(1000, self.start_cameras_sequence)
        self.window.after(1000, self.conectar_opc_thread)

    def start_cameras_sequence(self):
        for i, config in enumerate(CAMERAS_CONFIG):
            self.lbl_cams[i].config(text=f"CONECTANDO {config['desc']}...", fg="yellow")
            self.window.update() 
            try:
                stream = CameraStream(config).start()
                self.camera_streams.append(stream)
            except Exception as e:
                print(e)
                self.lbl_cams[i].config(text="ERROR", fg="red")
        self.video_loop()

    def video_loop(self):
        for i in range(len(self.camera_streams)):
            stream = self.camera_streams[i]
            ret, frame = stream.read()
            if ret and frame is not None:
                self.current_frames_raw[i] = frame
                frame_gui = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT)) 
                frame_rgb = cv2.cvtColor(frame_gui, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_cams[i].imgtk = imgtk 
                self.lbl_cams[i].configure(image=imgtk, text="") 
        self.window.after(30, self.video_loop)

    # =========================================================
    #  FASE 1: CAPTURA RÁPIDA (CON CREACIÓN DE FILA PREVIA)
    # =========================================================
    def capturar_botella(self):
        if self.contador_lote >= LIMITE_LOTE:
            messagebox.showwarning("Lote Completo", "Límite alcanzado. Debes PROCESAR el lote ahora.")
            return

        NUM_MUESTRAS = 5
        INTERVALO_MS = 0.05
        
        datos_botella = []
        timestamp_captura = datetime.now().strftime("%H:%M:%S")

        print(f"--- CAPTURANDO BOTELLA {self.contador_lote + 1} ---")
        
        for _ in range(NUM_MUESTRAS):
            frames_momento = [] 
            for stream in self.camera_streams:
                _, f = stream.read()
                if f is not None: frames_momento.append(f.copy()) 
                else: frames_momento.append(np.zeros((100,100,3), dtype=np.uint8))
            datos_botella.append(frames_momento)
            time.sleep(INTERVALO_MS)
        
        # --- CREAMOS LA FILA EN LA TABLA Y GUARDAMOS SU ID ---
        # Insertamos en posición 0 (arriba)
        row_values = [timestamp_captura, "ESPERA...", "...", "", ""]
        # El método insert retorna el 'iid' (identificador único de la fila)
        row_id = self.tree.insert("", 0, values=row_values, tags=("pendiente",))

        # Guardamos TODO en el buffer: Las fotos Y el ID de la fila para actualizarla luego
        self.buffer_lote.append({
            "frames": datos_botella,
            "row_id": row_id,
            "timestamp": timestamp_captura
        })
        
        self.contador_lote += 1
        self.btn_capturar.config(text=f"CAPTURAR BOTELLA ({self.contador_lote}/{LIMITE_LOTE})")

    # =========================================================
    #  FASE 2: PROCESAMIENTO (ACTUALIZANDO LAS FILAS EXISTENTES)
    # =========================================================
    def procesar_lote_completo(self, active_conf_map):
        if not self.buffer_lote:
            messagebox.showinfo("Vacío", "No hay fotos capturadas para procesar.")
            return
        
        self.btn_capturar.config(state="disabled")
        self.btn_procesar.config(state="disabled", text="PROCESANDO...")
        
        t = threading.Thread(target=lambda: self._logica_procesamiento(active_conf_map))
        t.start()

    def _logica_procesamiento(self, active_conf_map):
        total_botellas = len(self.buffer_lote)
        print(f"--- INICIANDO PROCESAMIENTO DE {total_botellas} BOTELLAS ---")
        
        # Iteramos sobre el buffer guardado
        for idx, item_data in enumerate(self.buffer_lote):
            rafaga_botella = item_data["frames"]
            current_row_id = item_data["row_id"] # ID de la fila en la tabla
            timestamp_origen = item_data["timestamp"]

            errores_consolidados = set()
            todas_las_clases_vistas = set()
            
            ultimos_frames_guardados = [None] * len(CAMERAS_CONFIG)
            ultimos_frames_anotados = [None] * len(CAMERAS_CONFIG)

            for frames_actuales in rafaga_botella:
                for i, frame in enumerate(frames_actuales):
                    frame_ann = frame.copy()
                    
                    results = self.model(frame, conf=0.1, verbose=False)[0]
                    detections = sv.Detections.from_ultralytics(results)

                    filter_mask = []
                    for class_id, score in zip(detections.class_id, detections.confidence):
                        nombre_clase = self.model.names[class_id]
                        umbral_requerido = active_conf_map.get(nombre_clase, DEFAULT_CONFIDENCE)
                        pasa_filtro = score >= umbral_requerido
                        filter_mask.append(pasa_filtro)

                    if len(filter_mask) > 0: detections = detections[np.array(filter_mask)]
                    else: detections = sv.Detections.empty()

                    labels = []
                    for k, class_id in enumerate(detections.class_id):
                        nombre_clase = self.model.names[class_id]
                        score = detections.confidence[k]
                        todas_las_clases_vistas.add(nombre_clase)
                        labels.append(f"{nombre_clase} {score:.0%}")
                        if nombre_clase in ACTIVE_ERRORS:
                            errores_consolidados.add(nombre_clase)

                    frame_ann = self.box_annotator.annotate(scene=frame_ann, detections=detections)
                    frame_ann = self.label_annotator.annotate(scene=frame_ann, detections=detections, labels=labels)
                    
                    ultimos_frames_guardados[i] = frame
                    ultimos_frames_anotados[i] = frame_ann

            if ENABLE_CUERPO:
                total_dobladas = self.stats_counters.get('cuerpo_doblada', 0)
                total_arrugadas = self.stats_counters.get('cuerpo_arrugada', 0)
                if 'cuerpo_doblada' in errores_consolidados or 'cuerpo_arrugada' in errores_consolidados:
                    errores_consolidados.add('cuerpo_arrugada_doblada')
            
            if ENABLE_CUERPO:
                vio_cuerpo_ok = "cuerpo_ok" in todas_las_clases_vistas
                hay_error_cuerpo = any("cuerpo" in err for err in errores_consolidados)
                if not hay_error_cuerpo and not vio_cuerpo_ok:
                    errores_consolidados.add("cuerpo_ausente")

            if ENABLE_CUELLO:
                vio_algo_cuello = any("cuello" in nombre for nombre in todas_las_clases_vistas)
                if not vio_algo_cuello: errores_consolidados.add("cuello_ausente")

            # --- GUARDADO EN DISCO ---
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Usamos el idx para asegurar orden en carpeta
            save_path = os.path.join(OUTPUT_FOLDER, f"evento_{timestamp_file}_b{idx+1}")
            global_result = "OK" if not errores_consolidados else "NOK"

            os.makedirs(save_path, exist_ok=True)
            for i, conf in enumerate(CAMERAS_CONFIG):
                cam_name = conf['desc']
                if ultimos_frames_guardados[i] is not None:
                    cv2.imwrite(os.path.join(save_path, f"{cam_name}_raw.jpg"), ultimos_frames_guardados[i])
                    cv2.imwrite(os.path.join(save_path, f"{cam_name}_ann.jpg"), ultimos_frames_anotados[i])

            if global_result == "OK": self.stats_counters['TOTAL_OK'] += 1
            else: 
                self.stats_counters['TOTAL_NOK'] += 1
                for err in errores_consolidados:
                    if err in self.stats_counters: self.stats_counters[err] += 1

            # --- ACTUALIZAR FILA EXISTENTE (USANDO 'item' NO 'insert') ---
            def update_ui_row():
                row_values = [timestamp_origen, global_result]
                for col in ACTIVE_ERRORS:
                    row_values.append("1" if col in errores_consolidados else "")
                row_values.append("ENV" if self.opc.connected else "-")

                tags = []
                if "ausente" in str(errores_consolidados): tags.append("warning") 
                elif global_result == "NOK": tags.append("nok")
                else: tags.append("ok")
                
                # AQUI ESTA LA CLAVE: ACTUALIZAMOS LA FILA CON EL ID GUARDADO
                try:
                    self.tree.item(current_row_id, values=row_values, tags=tuple(tags))
                except:
                    print("Error actualizando fila (tal vez fue borrada manual?)")
                
                # Actualizar totales
                total_vals = ["TOTALES", f"OK:{self.stats_counters['TOTAL_OK']} NOK:{self.stats_counters['TOTAL_NOK']}"]
                for col in ACTIVE_ERRORS:
                    count = self.stats_counters[col]
                    total_vals.append(str(count) if count > 0 else "")
                total_vals.append("-")
                self.tree.item("total_row", values=total_vals)
                
                self.btn_procesar.config(text=f"PROCESANDO ({idx+1}/{total_botellas})...")
            
            self.window.after(0, update_ui_row)
        
        self.buffer_lote = [] 
        
        def finish_ui():
            self.btn_capturar.config(state="normal", text=f"CAPTURAR BOTELLA ({self.contador_lote}/{LIMITE_LOTE})")
            self.btn_procesar.config(state="normal", text="PROCESAR LOTE COMPLETO")
            messagebox.showinfo("Fin", "Lote procesado exitosamente.")

        self.window.after(0, finish_ui)

    def conectar_opc_thread(self):
        t = threading.Thread(target=self._conectar_opc_logic)
        t.start()
    
    def _conectar_opc_logic(self):
        self.lbl_opc_status.config(text="Conectando...", fg="orange")
        self.btn_connect_opc.config(state="disabled")
        ok, msg = self.opc.conectar()
        if ok:
            self.lbl_opc_status.config(text=f"OPC ONLINE", fg="green")
            self.btn_connect_opc.config(text="Desconectar", command=self.desconectar_opc, state="normal")
        else:
            self.lbl_opc_status.config(text=f"OPC ERROR: {msg}", fg="red")
            self.btn_connect_opc.config(state="normal")
    
    def enviar_opc_manual(self):
        if not self.opc.connected:
            messagebox.showwarning("Error OPC", "No hay conexión OPC.")
            return

        respuesta = messagebox.askyesno("Confirmar Ciclo", "¿Desea enviar los datos actuales?")
        if not respuesta: return
        t = threading.Thread(target=self._rutina_envio_y_reset)
        t.start()

    def _rutina_envio_y_reset(self):
        print("--- INICIANDO CICLO OPC ---")
        self.opc.actualizar_tags(self.stats_counters)
        print(">> OPC: Enviando STATUS 128...")
        self.opc.escribir_tag_individual('STATUS', 128)
        print(">> OPC: Esperando...")
        for i in range(85):
            self.opc.actualizar_tags(self.stats_counters)
            self.opc.escribir_tag_individual('STATUS', 128)
            if not self.opc.connected: 
                print(">> OPC: Desconexión detectada. Abortando.")
                return
            time.sleep(1) 
        print(">> OPC: Reseteando contadores a 0...")
        diccionario_ceros = {k: 0 for k in OPC_TAGS_MAP.keys()}
        self.opc.actualizar_tags(diccionario_ceros)
        self.reset_local_counters()
        print("--- CICLO FINALIZADO ---")
        self.window.after(0, lambda: messagebox.showinfo("OPC", "ENVIADO.\nNuevo lote iniciado."))

    def reset_local_counters(self):
        self.stats_counters = {k: 0 for k in ACTIVE_ERRORS}
        self.stats_counters['TOTAL_OK'] = 0
        self.stats_counters['TOTAL_NOK'] = 0
        if 'cuerpo_arrugada_doblada' in self.stats_counters:
             self.stats_counters['cuerpo_arrugada_doblada'] = 0
        self.contador_lote = 0
        self.buffer_lote = [] 
        
        def update_gui():
            self.btn_capturar.config(text=f"CAPTURAR BOTELLA (0/{LIMITE_LOTE})")
            vals_iniciales = ["TOTALES", "OK:0 / NOK:0"] + ["0"]*len(ACTIVE_ERRORS) + ["-"]
            self.tree.item("total_row", values=vals_iniciales)
            for item in self.tree.get_children():
                if item != "total_row": self.tree.delete(item)

        self.window.after(0, update_gui)

    def desconectar_opc(self):
        self.opc.desconectar()
        self.lbl_opc_status.config(text="Desconectado", fg="black")
        self.btn_connect_opc.config(text="Conectar OPC", command=self.conectar_opc_thread)

    def on_closing(self):
        for stream in self.camera_streams:
            stream.release()
        self.opc.desconectar()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = QualityApp(root, "Packaging Bottle Inspector")
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()