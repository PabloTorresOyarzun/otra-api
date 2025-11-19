# -*- coding: utf-8 -*-
import fitz
import sys
import cv2
import numpy as np

# Constantes
UMBRAL_IMAGEN_RATIO = 0.7
UMBRAL_TEXTO_LARGO = 200
UMBRAL_TEXTO_CORTO = 200
RATIO_ASPECTO_VERTICAL = 1.5
RATIO_ASPECTO_HORIZONTAL = 1.5
RATIO_VERTICAL_ROTACION = 0.6
ANGULO_NORMAL = 0.5
ANGULO_LIGERO = 2.0
ANGULO_MEDIO = 10.0


def es_pagina_escaneada(pagina, umbral_imagen_ratio=UMBRAL_IMAGEN_RATIO):
    """Determina si una página es escaneada o digital."""
    area_pagina = abs(pagina.rect.width * pagina.rect.height)
    imagenes = pagina.get_images(full=True)
    texto = pagina.get_text().strip()
    
    if not imagenes:
        return False
    
    if len(texto) > UMBRAL_TEXTO_LARGO:
        return False
    
    area_imagenes = 0
    for img in imagenes:
        bbox = pagina.get_image_bbox(img)
        if bbox:
            area_imagenes += abs(bbox.width * bbox.height)
    
    ocupacion_imagen = area_imagenes / area_pagina if area_pagina > 0 else 0
    es_escaneada = ocupacion_imagen > umbral_imagen_ratio and len(texto) < UMBRAL_TEXTO_CORTO
    
    return es_escaneada


def detectar_orientacion_digital(pagina):
    """Detecta orientación en páginas digitales analizando dirección del texto."""
    text_dict = pagina.get_text("dict")
    directions = {"horizontal": 0, "vertical": 0}
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    bbox = span["bbox"]
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    
                    if height > width * RATIO_ASPECTO_VERTICAL:
                        directions["vertical"] += 1
                    elif width > height * RATIO_ASPECTO_HORIZONTAL:
                        directions["horizontal"] += 1
    
    total = sum(directions.values())
    if total == 0:
        return "SIN TEXTO"
    
    vertical_ratio = directions["vertical"] / total
    return "ROTADA" if vertical_ratio > RATIO_VERTICAL_ROTACION else "NORMAL"


def detectar_orientacion_escaneada(pagina, doc):
    """Detecta orientación en páginas escaneadas usando OpenCV."""
    try:
        imagenes = pagina.get_images(full=True)
        if not imagenes:
            return "SIN IMAGEN"
        
        img_ref = imagenes[0]
        base_img = doc.extract_image(img_ref[0])
        img_data = base_img["image"]
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return "ERROR AL DECODIFICAR"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return "NO SE DETECTARON LÍNEAS"
        
        angulos = []
        for line in lines:
            rho, theta = line[0]
            angulo_rad = theta - np.pi/2
            angulo_deg = np.degrees(angulo_rad)
            angulos.append(angulo_deg)
        
        angulos_horizontales = [a for a in angulos if abs(a) <= 45]
        
        if not angulos_horizontales:
            return "NO HAY LÍNEAS HORIZONTALES"
        
        angulo_promedio = np.mean(angulos_horizontales)
        
        if abs(angulo_promedio) < ANGULO_NORMAL:
            return "NORMAL"
        elif abs(angulo_promedio) < ANGULO_LIGERO:
            return f"LIGERAMENTE INCLINADA ({angulo_promedio:.1f}°)"
        elif abs(angulo_promedio) < ANGULO_MEDIO:
            return f"INCLINADA ({angulo_promedio:.1f}°)"
        else:
            return f"MUY INCLINADA ({angulo_promedio:.1f}°)"
            
    except Exception as e:
        return f"ERROR: {str(e)}"


def analizar_pdf_completo(doc):
    """Analiza todas las páginas de un PDF y retorna resultados."""
    resultados = []
    
    for i in range(len(doc)):
        pagina = doc[i]
        es_escaneada = es_pagina_escaneada(pagina)
        num_imagenes = len(pagina.get_images())
        rotacion_formal = pagina.rotation
        texto_length = len(pagina.get_text().strip())
        
        if es_escaneada:
            orientacion = detectar_orientacion_escaneada(pagina, doc)
        else:
            orientacion = detectar_orientacion_digital(pagina)
        
        resultados.append({
            'pagina': i + 1,
            'escaneada': es_escaneada,
            'num_imagenes': num_imagenes,
            'rotacion_formal': rotacion_formal,
            'orientacion': orientacion,
            'chars_texto': texto_length
        })
    
    return resultados


def paso_1_analizar_documento(doc):
    """PASO 1: Analiza orientación y tipo de páginas del documento."""
    print("=" * 60)
    print("PASO 1: ANÁLISIS DE DOCUMENTO")
    print("=" * 60)
    print()
    
    resultados = analizar_pdf_completo(doc)
    
    for r in resultados:
        tipo = "ESCANEADA" if r['escaneada'] else "DIGITAL"
        print(f"Página {r['pagina']}: {tipo} ({r['num_imagenes']} imágenes)")
        print(f"  Rotación: {r['rotacion_formal']}°")
        print(f"  Orientación: {r['orientacion']}")
        print()
    
    return resultados


def paso_2_corregir_rotacion(doc, resultados_paso_1):
    """PASO 2: Corrige la rotación de páginas rotadas."""
    print("=" * 60)
    print("PASO 2: CORRECCIÓN DE ROTACIÓN")
    print("=" * 60)
    print()
    
    paginas_corregidas = 0
    
    for resultado in resultados_paso_1:
        idx = resultado['pagina'] - 1
        pagina = doc[idx]
        
        # Corregir rotación formal si existe
        if resultado['rotacion_formal'] != 0:
            pagina.set_rotation(0)
            paginas_corregidas += 1
            print(f"Página {resultado['pagina']}: Rotación formal {resultado['rotacion_formal']}° corregida")
        
        # Corregir páginas digitales rotadas (texto vertical)
        elif not resultado['escaneada'] and resultado['orientacion'] == "ROTADA":
            # Rotar 90° en sentido horario para páginas con texto vertical
            pagina.set_rotation(270)
            paginas_corregidas += 1
            print(f"Página {resultado['pagina']}: Texto vertical corregido (rotación 270°)")
    
    print()
    if paginas_corregidas > 0:
        print(f"Total de páginas corregidas: {paginas_corregidas}")
    else:
        print("No se encontraron páginas que requieran corrección de rotación.")
    
    return paginas_corregidas



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python rotation.py <ruta_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error al abrir PDF: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # PASO 1: Análisis de documento
        resultados_paso_1 = paso_1_analizar_documento(doc)
        
        # PASO 2: Corrección de rotación
        paginas_corregidas = paso_2_corregir_rotacion(doc, resultados_paso_1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        doc.close()
        sys.exit(1)
    finally:
        doc.close()