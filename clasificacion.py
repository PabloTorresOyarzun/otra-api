import fitz
import httpx
import asyncio
from typing import List, Dict, Tuple
from patrones import PATRONES_INICIO, PATRON_DEFAULT
from config import get_settings, calcular_timeout_azure


settings = get_settings()


def recortar_header(pdf_bytes: bytes) -> bytes:
    """Recorta el 30% superior de cada página del PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    new_doc = fitz.open()
    
    for page in doc:
        rect = page.rect
        crop_rect = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * 0.3)
        page.set_cropbox(crop_rect)
        new_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
    
    pdf_recortado = new_doc.tobytes()
    new_doc.close()
    doc.close()
    
    return pdf_recortado


async def extraer_texto_documento_completo(
    pdf_bytes: bytes, 
    azure_endpoint: str = "http://azure-di-layout:5000"
) -> Tuple[Dict[int, str], str]:
    """
    Extrae texto de TODAS las páginas de un PDF usando Azure Document Intelligence.
    Retorna (dict_paginas, estado) donde:
    - dict_paginas: {numero_pagina: texto_extraido}
    - estado: indicador de éxito o error
    """
    try:
        # Calcular timeout basado en número de páginas
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_paginas = len(doc)
        doc.close()
        
        timeout_total = calcular_timeout_azure(num_paginas)
        max_attempts = int(timeout_total / 2)  # Poll cada 2 segundos
        
        url = f"{azure_endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30"
        
        headers = {
            "Content-Type": "application/pdf"
        }
        
        timeout_config = httpx.Timeout(
            connect=settings.TIMEOUT_CONNECT,
            read=timeout_total,
            write=timeout_total,
            pool=5.0
        )
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(url, headers=headers, content=pdf_bytes)
            response.raise_for_status()
            
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                return {}, "error_no_operation_location"
            
            # Polling con backoff exponencial
            for attempt in range(max_attempts):
                wait_time = min(1 * (1.5 ** attempt), 5)  # Max 5s entre polls
                await asyncio.sleep(wait_time)
                
                result_response = await client.get(operation_location)
                result_data = result_response.json()
                
                if result_data.get("status") == "succeeded":
                    texto_por_pagina = {}
                    
                    if "analyzeResult" in result_data:
                        pages = result_data["analyzeResult"].get("pages", [])
                        for page in pages:
                            page_number = page.get("pageNumber", 0)
                            texto_pagina = ""
                            
                            for line in page.get("lines", []):
                                texto_pagina += line.get("content", "") + " "
                            
                            texto_por_pagina[page_number] = texto_pagina.strip()
                    
                    return texto_por_pagina, "success"
                
                elif result_data.get("status") in ["failed", "invalid"]:
                    return {}, f"error_azure_status_{result_data.get('status')}"
            
            return {}, "error_timeout"
        
    except httpx.TimeoutException:
        return {}, "error_timeout"
    except Exception as e:
        return {}, f"error_exception_{type(e).__name__}"


def clasificar_pagina(texto: str) -> str:
    """
    Clasifica una página buscando la aparición más temprana de cualquier
    patrón de documento en el texto.
    """
    texto_upper = texto.upper()
    matches = []
    
    for tipo_documento, patrones in PATRONES_INICIO.items():
        for patron in patrones:
            index = texto_upper.find(patron.upper())
            
            if index != -1:
                matches.append({
                    "index": index,
                    "tipo": tipo_documento
                })
    
    if not matches:
        return PATRON_DEFAULT
    
    matches.sort(key=lambda x: x["index"])
    return matches[0]["tipo"]


async def clasificar_documento_completo(pdf_bytes: bytes) -> List[Dict]:
    """
    Clasifica todas las páginas de un PDF:
    1. Recorta el 30% superior de todas las páginas
    2. Envía el PDF completo recortado a Azure DI (una sola llamada)
    3. Clasifica cada página según el texto extraído
    
    Retorna lista con clasificación por página.
    """
    pdf_recortado = recortar_header(pdf_bytes)
    texto_por_pagina, _ = await extraer_texto_documento_completo(pdf_recortado)
    
    clasificaciones = []
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_paginas = len(doc)
    doc.close()
    
    for num_pagina in range(1, total_paginas + 1):
        texto_pagina = texto_por_pagina.get(num_pagina, "")
        tipo_documento = clasificar_pagina(texto_pagina)
        
        clasificaciones.append({
            "pagina": num_pagina,
            "tipo": tipo_documento
        })
    
    return clasificaciones


def segmentar_pdf(pdf_bytes: bytes, clasificaciones: List[Dict]) -> List[Dict]:
    """
    Segmenta un PDF en múltiples documentos según las clasificaciones.
    Retorna lista de documentos segmentados con sus metadatos.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    documentos_segmentados = []
    
    paginas_por_tipo = {}
    tipo_actual = None
    inicio_segmento = 0
    
    for i, clasificacion in enumerate(clasificaciones):
        tipo = clasificacion["tipo"]
        
        if tipo != tipo_actual and tipo != PATRON_DEFAULT:
            if tipo_actual is not None:
                paginas_por_tipo[f"{tipo_actual}_{inicio_segmento}"] = {
                    "tipo": tipo_actual,
                    "inicio": inicio_segmento,
                    "fin": i - 1
                }
            tipo_actual = tipo
            inicio_segmento = i
        elif tipo == PATRON_DEFAULT and tipo_actual is not None:
            continue
    
    if tipo_actual is not None:
        paginas_por_tipo[f"{tipo_actual}_{inicio_segmento}"] = {
            "tipo": tipo_actual,
            "inicio": inicio_segmento,
            "fin": len(clasificaciones) - 1
        }
    
    if not paginas_por_tipo:
        nuevo_doc = fitz.open()
        nuevo_doc.insert_pdf(doc)
        documentos_segmentados.append({
            "tipo": PATRON_DEFAULT,
            "paginas": list(range(1, len(doc) + 1)),
            "pdf_bytes": nuevo_doc.tobytes()
        })
        nuevo_doc.close()
    else:
        for key, segmento in paginas_por_tipo.items():
            nuevo_doc = fitz.open()
            nuevo_doc.insert_pdf(doc, from_page=segmento["inicio"], to_page=segmento["fin"])
            
            documentos_segmentados.append({
                "tipo": segmento["tipo"],
                "paginas": list(range(segmento["inicio"] + 1, segmento["fin"] + 2)),
                "pdf_bytes": nuevo_doc.tobytes()
            })
            nuevo_doc.close()
    
    doc.close()
    return documentos_segmentados