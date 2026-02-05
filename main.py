import cv2
import time
import os
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
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
  { "id": 2, "desc": "CAM_DER", "foco": 228 },
  { "id": 4, "desc": "CAM_CEN", "foco": 45 },
  { "id": 0, "desc": "CAM_IZQ", "foco": 228 }
]

MODEL_PATH = './rtdetr-x_4.pt' 
OUTPUT_FOLDER = "registro_inspecciones"
OPC_URL = "opc.tcp://172.16.40.150:49340"

# ==========================================
#    CONFIGURACIÓN DE SENSIBILIDAD
# ==========================================
DEFAULT_CONFIDENCE = 0.25

CONFIDENCE_MAP_1 = {
  'cuerpo_alreves': 0.9,
  'cuerpo_arrugada': 0.7,
  'cuerpo_ausente': 0.5,
  'cuerpo_doblada': 0.9,
  'cuerpo_falla_adherencia': 0.8,
  'cuerpo_invertida': 0.9,
  'cuerpo_ok': 0.3,
  'cuerpo_multiple': 0.8,
  'cuerpo_rasgada': 0.9,
  'cuerpo_sucia': 0.9,
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
    
    try: self.model = RTDETR(MODEL_PATH)
    except: 
      print("⚠️ Usando modelo standard.")
      self.model = RTDETR('rtdetr-l.pt')
    
    self.box_annotator = sv.BoxAnnotator(thickness=2)
    self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
    self.opc = OPCClient(OPC_URL)
    
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

    self.frame_control = tk.Frame(window, bg="#ecf0f1")
    self.frame_control.pack(fill="x", pady=5)

    info_text = "CONFIGURACIÓN ACTIVA: "
    info_text += "[✅ CUELLO] " if ENABLE_CUELLO else "[❌ CUELLO] "
    info_text += "[✅ CUERPO]" if ENABLE_CUERPO else "[❌ CUERPO]"
    tk.Label(self.frame_control, text=info_text, bg="#ecf0f1", fg="#7f8c8d", font=("Arial", 11)).pack(pady=2)

    self.frame_buttons = tk.Frame(self.frame_control, bg="#ecf0f1")
    self.frame_buttons.pack(pady=5)

    # --- BOTÓN 1: ANALIZAR UNA BOTELLA ---
    self.btn_analizar = tk.Button(self.frame_buttons, 
                    text="ANALIZAR BOTELLA (0)",
                    command=self.analizar_una_botella_click, 
                    bg="#d35400", fg="white",
                    font=("Arial", 16, "bold"), height=2, width=30)
    self.btn_analizar.pack(side="left", padx=10)

    # --- BOTÓN 2: ENVIAR A ATHENA ---
    self.btn_enviar = tk.Button(self.frame_buttons, 
                    text="ENVIAR A ATHENA",
                    command=self.enviar_a_athena_click, 
                    bg="#27ae60", fg="white",
                    font=("Arial", 16, "bold"), height=2, width=30)
    self.btn_enviar.pack(side="left", padx=10)

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
    self.tree.tag_configure("pendiente", background="#bdc3c7", foreground="#7f8c8d") 

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

  # ========================================================
  #   LÓGICA BOTÓN 1: ANALIZAR BOTELLA INMEDIATAMENTE
  # ========================================================
  def analizar_una_botella_click(self):
    if self.contador_lote >= LIMITE_LOTE:
       messagebox.showwarning("Límite", "Has llegado a 24. Envía a Athena o Reinicia.")
       return

    # 1. Tomar fotos (pequeño delay para asegurar ráfaga)
    NUM_MUESTRAS = 3
    INTERVALO_MS = 0.1
    rafaga_frames = []
    
    timestamp_captura = datetime.now().strftime("%H:%M:%S")

    # Captura Ráfaga
    for _ in range(NUM_MUESTRAS):
      frames_momento = [] 
      for stream in self.camera_streams:
        _, f = stream.read()
        if f is not None: frames_momento.append(f.copy()) 
        else: frames_momento.append(np.zeros((100,100,3), dtype=np.uint8))
      rafaga_frames.append(frames_momento)
      time.sleep(INTERVALO_MS)
    
    # 2. Crear fila "Pendiente" en UI
    row_values = [timestamp_captura, "ANALIZANDO...", "...", "", ""]
    row_id = self.tree.insert("", 0, values=row_values, tags=("pendiente",))

    # 3. Lanzar hilo de análisis (No bloquea la cámara)
    t = threading.Thread(target=self._thread_analisis_individual, args=(rafaga_frames, row_id, timestamp_captura))
    t.start()
    
    self.contador_lote += 1
    self.btn_analizar.config(text=f"ANALIZAR BOTELLA ({self.contador_lote})")

  def _thread_analisis_individual(self, rafaga_frames, row_id, timestamp_origen):
      errores_consolidados = set()
      todas_las_clases_vistas = set()
      
      ultimos_frames_guardados = [None] * len(CAMERAS_CONFIG)
      ultimos_frames_anotados = [None] * len(CAMERAS_CONFIG)

      # --- INFERENCIA IA ---
      for frames_actuales in rafaga_frames:
        for i, frame in enumerate(frames_actuales):
          frame_ann = frame.copy()
          
          results = self.model(frame, conf=0.1, verbose=False)[0]
          detections = sv.Detections.from_ultralytics(results)

          filter_mask = []
          for class_id, score in zip(detections.class_id, detections.confidence):
            nombre_clase = self.model.names[class_id]
            umbral_requerido = CONFIDENCE_MAP_1.get(nombre_clase, DEFAULT_CONFIDENCE)
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

      # --- LÓGICA DE NEGOCIO ---
      if ENABLE_CUERPO:
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
      # Usamos el contador lote actual para el nombre de carpeta
      save_path = os.path.join(OUTPUT_FOLDER, f"evento_{timestamp_file}_b{self.contador_lote}")
      global_result = "OK" if not errores_consolidados else "NOK"

      try:
        os.makedirs(save_path, exist_ok=True)
        for i, conf in enumerate(CAMERAS_CONFIG):
            cam_name = conf['desc']
            if ultimos_frames_guardados[i] is not None:
                cv2.imwrite(os.path.join(save_path, f"{cam_name}_raw.jpg"), ultimos_frames_guardados[i])
                cv2.imwrite(os.path.join(save_path, f"{cam_name}_ann.jpg"), ultimos_frames_anotados[i])
      except Exception as e:
        print(f"Error guardando imagenes: {e}")

      # --- ACTUALIZAR CONTADORES GLOBALES ---
      if global_result == "OK": self.stats_counters['TOTAL_OK'] += 1
      else: 
        self.stats_counters['TOTAL_NOK'] += 1
        for err in errores_consolidados:
          if err in self.stats_counters: self.stats_counters[err] += 1

      # --- PREPARAR DATOS PARA LA UI ---
      row_values = [timestamp_origen, global_result]
      for col in ACTIVE_ERRORS:
        row_values.append("1" if col in errores_consolidados else "")
      row_values.append("-") # OPC aun no enviado

      tags = []
      if "ausente" in str(errores_consolidados): tags.append("warning") 
      elif global_result == "NOK": tags.append("nok")
      else: tags.append("ok")
      
      # Llamada segura a Tkinter desde el hilo
      self.window.after(0, self.actualizar_fila_safe, row_id, row_values, tuple(tags))

  def actualizar_fila_safe(self, row_id, values, tags):
      try:
          self.tree.item(row_id, values=values, tags=tags)
      except: pass
      
      # Actualizar totales
      total_vals = ["TOTALES", f"OK:{self.stats_counters['TOTAL_OK']} NOK:{self.stats_counters['TOTAL_NOK']}"]
      for col in ACTIVE_ERRORS:
        count = self.stats_counters[col]
        total_vals.append(str(count) if count > 0 else "")
      total_vals.append("-")
      self.tree.item("total_row", values=total_vals)

  # ========================================================
  #   LÓGICA BOTÓN 2: ENVIAR A ATHENA (Cualquier cantidad)
  # ========================================================
  def enviar_a_athena_click(self):
    if not self.opc.connected:
       messagebox.showerror("Error", "OPC No conectado.")
       return
    
    if self.contador_lote == 0:
        if not messagebox.askyesno("Vacío", "No hay botellas analizadas. ¿Enviar ceros?"):
            return

    self.btn_analizar.config(state="disabled")
    self.btn_enviar.config(state="disabled", text="ENVIANDO...")
    
    t = threading.Thread(target=self._thread_envio_opc)
    t.start()

  def _thread_envio_opc(self):
        def actualizar_btn_texto(txt):
            self.btn_enviar.config(text=txt)

        print("--- INICIANDO ENVIO A ATHENA ---")
        self.opc.actualizar_tags(self.stats_counters)
        
        print(">> OPC: Enviando STATUS 128...")
        self.opc.escribir_tag_individual('STATUS', 128)
        
        # Actualizar UI para mostrar que se envió (Columna OPC)
        self.window.after(0, self._marcar_filas_como_enviadas)

        print(">> OPC: Esperando ciclo de máquina (120s)...")
        for i in range(120):
            self.opc.actualizar_tags(self.stats_counters)
            self.opc.escribir_tag_individual('STATUS', 128)
            msg_espera = f"ENVIANDO... ({120-i}s)"
            self.window.after(0, lambda m=msg_espera: actualizar_btn_texto(m))
            
            if not self.opc.connected: 
                print(">> OPC: Desconexión detectada.")
                break
            time.sleep(1) 
            
        print(">> OPC: Reseteando contadores...")
        diccionario_ceros = {k: 0 for k in OPC_TAGS_MAP.keys()}
        self.opc.actualizar_tags(diccionario_ceros)
        
        self.window.after(0, self.reset_local_counters)
        self.window.after(0, lambda: messagebox.showinfo("Ciclo Completado", "Datos enviados a Athena correctamente."))

  def _marcar_filas_como_enviadas(self):
      # Pone "ENV" en la columna final de todas las filas
      for item in self.tree.get_children():
          if item == "total_row": continue
          vals = list(self.tree.item(item, "values"))
          vals[-1] = "ENV"
          self.tree.item(item, values=vals)

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

  def reset_local_counters(self):
    self.stats_counters = {k: 0 for k in ACTIVE_ERRORS}
    self.stats_counters['TOTAL_OK'] = 0
    self.stats_counters['TOTAL_NOK'] = 0
    if 'cuerpo_arrugada_doblada' in self.stats_counters:
       self.stats_counters['cuerpo_arrugada_doblada'] = 0
    self.contador_lote = 0
    
    self.btn_analizar.config(state="normal", text=f"ANALIZAR BOTELLA (0)")
    self.btn_enviar.config(state="normal", text="ENVIAR A ATHENA")
    
    vals_iniciales = ["TOTALES", "OK:0 / NOK:0"] + ["0"]*len(ACTIVE_ERRORS) + ["-"]
    self.tree.item("total_row", values=vals_iniciales)
    # Borramos las filas de las botellas
    for item in self.tree.get_children():
      if item != "total_row": self.tree.delete(item)

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
