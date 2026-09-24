1980 interfaces en 55 bloques de evaluación --> 62 usuarios

&#x09;495 interfaces de pagina web

&#x09;495 interfaces de escritorio

&#x09;495 interfaces móviles

&#x09;495 pósters



CONTENIDO DE LA CARPETA DEL DATASET:

* Archivo image\_types.csv: Archivo que identifica identificador\_imagen | tipo\_imagen | bloque\_al\_que\_pertenece | Test/Train
* Carpeta images: contiene capturas de pantalla de todas las interfaces del dataset
* Carpeta eyetracker\_logs: Contiene los logs crudos del eye tracker. Archivo XX\_hk0YY\_fixations.csv se refiere al participante YY mirando el bloque XX
* Carpeta saliency\_maps: Contiene fixmaps, heatmaps y overlay\_heatmaps de diferentes tiempos. 1, 3 y 7 segundos.
* Carpeta scanpaths: Contiene scanpaths de diferentes duraciones organizados por id\_imagen donde los archivos de dentro N.png muestran el scanpath del participante N mirando la UI con id\_imagen





EXPLICACION FIXMAPS, HEATMAPS, OVERLAY HEATMAPS y SCANPATHS:



Fixmaps (Mapas de fijaciones): Son representaciones binarias o discretas (una matriz de ceros y unos) del estímulo visual. En un fixmap, los píxeles exactos donde el usuario ha fijado la mirada tienen valor 1 (o blanco), y el resto del espacio tiene valor 0 (o negro). Se utilizan como la "verdad absoluta" (ground truth) matemática para entrenar modelos predictivos de saliencia visual, ya que indican el punto exacto sin difuminar.



Heatmaps (Mapas de calor): Son representaciones continuas que muestran la densidad espacial de la atención. Se construyen aplicando un filtro gaussiano (un desenfoque matemático) sobre los fixmaps. Las zonas más observadas se representan con colores "calientes" (rojo, amarillo) y las menos observadas con colores "fríos" (azul, verde). Muestran el volumen de atención general, pero pierden el detalle del píxel exacto y el orden temporal.



Overlay Heatmaps (Mapas de calor superpuestos): Es el resultado de fusionar visualmente un heatmap con la imagen original de la interfaz (el estímulo). Se aplica transparencia (canal alfa) al mapa de calor para que puedas ver el diseño web o móvil de fondo y entender exactamente qué elementos gráficos (botones, textos, imágenes) están recibiendo esa carga visual.



Scanpaths (Rutas de escaneo): Es la única visualización que incorpora la dimensión temporal de forma explícita. Representa el comportamiento visual como una secuencia cronológica. Las fijaciones se dibujan como círculos (donde el tamaño del círculo suele ser proporcional a la duración de la fijación) y las sacadas se dibujan como líneas o flechas que conectan los círculos en orden (1, 2, 3...). Permite analizar la estrategia de exploración del usuario.



CONTENIDO eyetracker\_logs: https://www.gazept.com/dl/Gazepoint\_API\_v2.0.pdf



* Columnas MEDIA\_ID, MEDIA\_NAME: Contiene el identificador de la imagen, más útil el MEDIA\_NAME al ser coherente con el resto del data set
* CNT: The counter data variable is incremented by 1 for each data record sent by the server. Useful to determine if any data packets are missed by the client. INDEPENDIENTE PARA CADA MEDIA\_NAME
* TIME: Indica el tiempo de cada registro 1-7 segundos. INDEPENDIENTE PARA CADA MEDIA\_NAME
* TIME\_TICK: Sirve para conseguir más precisión en los tiempos
* FPOGX y FPOGY: The X- and Y-coordinates of the fixation POG, as a fraction of the screen size.(0,0) is top left, (0.5,0.5) is the screen center, and (1.0,1.0) is bottom right.
* FPOGS:  The starting time of the fixation POG in seconds since the system initialization or calibration.
* FPOGD: The duration of the fixation POG in seconds.
* FPOGID:  The fixation POG ID number
* FPOGV: The valid flag with value of 1 if the fixation POG data is valid, and 0 if it is not.
* BPOGX, BPOGY: The X- and Y-coordinates of the best eye POG, as a fraction of the screen size.
* BPOGV: The valid flag with value of 1 if the data is valid, and 0 if it is not.
* CX, CY: The X- and Y-coordinates of the mouse cursor, as percentage of the screen size.
* CS: Mouse cursor state, 0 for steady state, 1 for left button down, 2 for right button down.
* USER: COLUMNA VACIA
* LPCX, LCPY: The X- and Y-coordinates of the left eye pupil in the camera image, as a fraction of the camera image size.
* LPD: The diameter of the left eye pupil in pixels.
* LPS:  The scale factor of the left eye pupil (unitless). Value equals 1 at calibration depth, is less than 1 when user is closer to the eye tracker and greater than 1 when user is further away.
* LPV: The valid flag with value of 1 if the data is valid, and 0 if it is not.
* RPCX, RPCY: The X- and Y-coordinates of the right eye pupil in the camera image, as a fraction of the camera image size.
* RPD: The diameter of the right eye pupil in pixels.
* RPS: The scale factor of the right eye pupil (unitless). Value equals 1 at calibration depth, is less than 1 when user is closer to the eye tracker and greater than 1 when user is further away
* RPV: The valid flag with value of 1 if the data is valid, and 0 if it is not.
* BKID,BKDUR,BKPMIN: NO SALE EN EL DOCUMENTO ASOCIADO. (Parpadeos /blinks)
* LPMM y LPMMV: Diámetro exacto de la pupila izquierda junto con el validador.
* RPMM y RPMMV: Diámetro exacto de la pupila derecha junto con el validador.
* DIAL, DIALV, GSR, GSRV, HR, HRV, HRP, TTL0, TTL1, TTLV, PIXS, PIXV, AOI: SIN VALORES 
* SACCADE\_MAG: La amplitud o tamaño del movimiento rápido del ojo (sacada) justo antes de esta fijación. Un valor alto (saltos largos) indica exploración global de la interfaz; un valor bajo indica procesamiento local o lectura.
* SACCADE\_DIR: El ángulo hacia el que saltó el ojo, medido en grados (de 0 a 360). Te permite analizar estrategias de escaneo visual (ej. movimientos horizontales dominantes durante la lectura de texto).
* VID\_FRAME: VACIO EN EL DATASET

