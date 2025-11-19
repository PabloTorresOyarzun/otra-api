from fastapi import FastAPI, HTTPException, APIRouter, File, UploadFile, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import os
import base64
import io
import fitz
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from quality import paso_1_analizar_documento, paso_2_corregir_rotacion
from clasificacion import clasificar_documento_completo, segmentar_pdf
from config import get_settings, calcular_timeout_excel, get_valid_api_tokens
from token_manager import token_manager


@contextmanager
def suprimir_prints():
    """Suprime temporalmente la salida a stdout."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = original_stdout


settings = get_settings()
app = FastAPI(title="API Docs")

# Esquema de seguridad para autenticación con token
security = HTTPBearer()


def verify_api_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifica que el token API proporcionado sea válido.

    Uso en Postman:
    - Header: Authorization
    - Value: Bearer <tu-token-aqui>
    """
    # Verificar desde el gestor de tokens (modo persistente)
    if token_manager.is_valid_token(credentials.credentials):
        return credentials.credentials

    # Fallback: verificar desde API_TOKENS en .env (modo legacy)
    valid_tokens = get_valid_api_tokens()
    if valid_tokens and credentials.credentials in valid_tokens:
        return credentials.credentials

    raise HTTPException(
        status_code=401,
        detail="Token de autenticación inválido"
    )


def verify_admin_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifica que el token de administrador sea válido.
    Solo permite acceso a endpoints de gestión de tokens.

    Uso en Postman:
    - Header: Authorization
    - Value: Bearer <admin-token>
    """
    if not settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="ADMIN_TOKEN no configurado en el servidor"
        )

    if credentials.credentials != settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=403,
            detail="Se requiere token de administrador"
        )

    return credentials.credentials


BASE_URL = "https://backend.juanleon.cl"
ENDPOINT_DESPACHO = "/api/admin/despachos/{codigo}"
ENDPOINT_DOCUMENTOS = "/api/admin/documentos64/despacho/{codigo_visible}"

# Almacenamiento en memoria
documentos_finales_sgd: Dict[str, List[Dict]] = {}
documentos_finales_individuales: Dict[str, List[Dict]] = {}

# Thread pool para operaciones CPU-bound
executor = ThreadPoolExecutor(max_workers=4)


class DocumentoSimplificado(BaseModel):
    nombre: str
    estado: str
    fecha_recepcion: str


class Usuarios(BaseModel):
    pedidor: Optional[List[str]] = None
    jefe_operaciones: Optional[List[str]] = None


class ConsultaResponse(BaseModel):
    codigo_despacho: str
    id_interno: str
    cliente: str
    estado: str
    tipo: str
    total_documentos: int
    documentos: List[DocumentoSimplificado]
    usuarios: Usuarios


class Alerta(BaseModel):
    pagina: int
    tipo: str
    descripcion: str


class DocumentoFinal(BaseModel):
    archivo_origen: str
    nombre_salida: str
    tipo: str
    paginas: List[int]
    alertas: Optional[List[Alerta]] = None


class ProcesamientoResponse(BaseModel):
    codigo_despacho: str
    cliente: str
    estado: str
    tipo: str
    total_documentos_segmentados: int
    documentos: List[DocumentoFinal]


class ProcesamientoIndividualResponse(BaseModel):
    archivo_origen: str
    total_documentos_segmentados: int
    documentos: List[DocumentoFinal]


class TokenInfo(BaseModel):
    id: str
    name: str
    masked_token: str
    created_at: str
    created_by: str
    last_used: Optional[str] = None
    is_active: bool = True


class TokenCreateRequest(BaseModel):
    name: str


class TokenCreateResponse(BaseModel):
    id: str
    token: str
    name: str
    created_at: str
    message: str


class TokenDeleteResponse(BaseModel):
    success: bool
    message: str


# Routers
sgd_router = APIRouter(prefix="/sgd", tags=["SGD"])
documentos_router = APIRouter(prefix="/documentos", tags=["Documentos"])
admin_router = APIRouter(prefix="/admin/tokens", tags=["Admin - Gestión de Tokens"])


def validar_pdf(file_bytes: bytes) -> bool:
    """Valida que los bytes sean un PDF válido."""
    if not file_bytes.startswith(b'%PDF'):
        return False
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        doc.close()
        return True
    except Exception:
        return False


def validar_excel(file_bytes: bytes, nombre_archivo: str) -> bool:
    """Valida que los bytes sean un archivo Excel válido."""
    extension = os.path.splitext(nombre_archivo.lower())[1]
    
    if extension in ['.xlsx', '.xlsm', '.xltx', '.xltm']:
        if not file_bytes.startswith(b'PK\x03\x04'):
            return False
    elif extension in ['.xls', '.xlsb']:
        if not file_bytes.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
            return False
    
    try:
        engine = 'xlrd' if extension == '.xls' else 'openpyxl'
        pd.read_excel(io.BytesIO(file_bytes), engine=engine, nrows=0)
        return True
    except Exception:
        return False


def validar_tamano_archivo(file_bytes: bytes) -> None:
    """Valida que el archivo no exceda el tamaño máximo."""
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo excede el tamaño máximo de {settings.MAX_FILE_SIZE_MB}MB"
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
)
async def consultar_despacho_detalle(codigo_interno: str, token: str) -> Optional[Dict]:
    if not token:
        return None
    
    url = f"{BASE_URL}{ENDPOINT_DESPACHO.format(codigo=str(codigo_interno))}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    timeout = httpx.Timeout(
        connect=settings.TIMEOUT_CONNECT,
        read=settings.TIMEOUT_READ,
        write=settings.TIMEOUT_WRITE,
        pool=5.0
    )
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            if 'application/json' not in response.headers.get('Content-Type', ''):
                return None
            
            datos = response.json()
            return datos if isinstance(datos, dict) else None
    except (httpx.TimeoutException, httpx.NetworkError):
        raise
    except Exception:
        return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError))
)
async def consultar_documentacion(codigo: str, token: str) -> Optional[List]:
    if not token:
        return None
    
    url = f"{BASE_URL}{ENDPOINT_DOCUMENTOS.format(codigo_visible=str(codigo))}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    timeout = httpx.Timeout(
        connect=settings.TIMEOUT_CONNECT,
        read=settings.TIMEOUT_READ,
        write=settings.TIMEOUT_WRITE,
        pool=5.0
    )
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            if 'application/json' not in response.headers.get('Content-Type', ''):
                return None
            
            datos_json = response.json()
            if not isinstance(datos_json, dict):
                return None
            
            documentos = datos_json.get("data", [])
            return documentos if isinstance(documentos, list) else None
    except (httpx.TimeoutException, httpx.NetworkError):
        raise
    except Exception:
        return None


def es_archivo_excel(nombre_archivo: str) -> bool:
    """Verifica si un archivo es Excel basándose en su extensión."""
    extensiones_excel = ['.xls', '.xlsx', '.xlsm', '.xlsb', '.xltx', '.xltm']
    extension = os.path.splitext(nombre_archivo.lower())[1]
    return extension in extensiones_excel


def _convertir_excel_a_pdf_sync(excel_bytes: bytes, nombre_archivo: str) -> bytes:
    """Función síncrona interna para conversión Excel a PDF."""
    try:
        extension = os.path.splitext(nombre_archivo.lower())[1]
        
        if extension == '.xls':
            engine = 'xlrd'
        else:
            engine = 'openpyxl'
        
        df_dict = pd.read_excel(
            io.BytesIO(excel_bytes),
            sheet_name=None,
            engine=engine
        )
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        for sheet_name, df in df_dict.items():
            title = Paragraph(f"<b>{sheet_name}</b>", styles['Heading1'])
            elements.append(title)
            elements.append(Spacer(1, 0.2*inch))
            
            df_clean = df.fillna('')
            data = [df_clean.columns.tolist()] + df_clean.values.tolist()
            data = [[str(cell) for cell in row] for row in data]
            
            page_width = A4[0] - inch
            num_cols = len(data[0]) if data else 1
            col_width = page_width / num_cols
            col_width = min(col_width, 2*inch)
            
            table = Table(data, colWidths=[col_width] * num_cols)
            
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            elements.append(table)
            elements.append(PageBreak())
        
        doc.build(elements)
        return buffer.getvalue()
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al convertir Excel a PDF: {str(e)}"
        )


async def convertir_excel_a_pdf(excel_bytes: bytes, nombre_archivo: str) -> bytes:
    """Convierte un archivo Excel a PDF de forma asíncrona con timeout dinámico."""
    timeout = calcular_timeout_excel(len(excel_bytes))
    loop = asyncio.get_event_loop()
    
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _convertir_excel_a_pdf_sync,
                excel_bytes,
                nombre_archivo
            ),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=408,
            detail=f"Conversión Excel excedió el tiempo límite de {timeout}s"
        )


def _procesar_calidad_sync(pdf_bytes: bytes) -> tuple:
    """Función síncrona interna para procesamiento de calidad."""
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    with suprimir_prints():
        resultados_paso_1 = paso_1_analizar_documento(pdf_doc)
        paginas_corregidas = paso_2_corregir_rotacion(pdf_doc, resultados_paso_1)
    
    if paginas_corregidas > 0:
        pdf_bytes = pdf_doc.tobytes()
    
    pdf_doc.close()
    return pdf_bytes, resultados_paso_1


async def procesar_pdf_completo(pdf_bytes: bytes, nombre_archivo: str) -> Dict[str, Any]:
    """
    Flujo completo de procesamiento de PDF:
    1. Análisis y corrección de calidad
    2. Clasificación de páginas
    3. Segmentación de documentos
    
    Retorna documentos finales segmentados con alertas de calidad.
    """
    try:
        loop = asyncio.get_event_loop()
        pdf_bytes, resultados_paso_1 = await loop.run_in_executor(
            executor,
            _procesar_calidad_sync,
            pdf_bytes
        )
        
        clasificaciones = await clasificar_documento_completo(pdf_bytes)
        
        documentos_segmentados = await loop.run_in_executor(
            executor,
            segmentar_pdf,
            pdf_bytes,
            clasificaciones
        )
        
        alertas_por_documento = {}
        for resultado in resultados_paso_1:
            alertas = []
            
            if resultado['escaneada']:
                if 'INCLINADA' in resultado['orientacion']:
                    alertas.append({
                        "pagina": resultado['pagina'],
                        "tipo": "inclinado",
                        "descripcion": f"Página escaneada {resultado['orientacion']}"
                    })
                elif resultado['orientacion'] not in ['NORMAL', 'SIN TEXTO', 'SIN IMAGEN']:
                    alertas.append({
                        "pagina": resultado['pagina'],
                        "tipo": "escaneado",
                        "descripcion": f"Página escaneada: {resultado['orientacion']}"
                    })
            
            if resultado['rotacion_formal'] != 0:
                alertas.append({
                    "pagina": resultado['pagina'],
                    "tipo": "rotado",
                    "descripcion": f"Rotación de {resultado['rotacion_formal']}° corregida"
                })
            
            if not resultado['escaneada'] and resultado['orientacion'] == 'ROTADA':
                alertas.append({
                    "pagina": resultado['pagina'],
                    "tipo": "rotado",
                    "descripcion": "Texto vertical corregido"
                })
            
            if alertas:
                alertas_por_documento[resultado['pagina']] = alertas
        
        documentos_finales = []
        for idx, doc_seg in enumerate(documentos_segmentados):
            alertas_segmento = []
            for pagina in doc_seg['paginas']:
                if pagina in alertas_por_documento:
                    alertas_segmento.extend(alertas_por_documento[pagina])
            
            nombre_salida = f"{nombre_archivo.replace('.pdf', '')}_{doc_seg['tipo']}_{idx+1}.pdf"
            
            documentos_finales.append({
                "archivo_origen": nombre_archivo,
                "nombre_salida": nombre_salida,
                "tipo": doc_seg['tipo'],
                "paginas": doc_seg['paginas'],
                "pdf_bytes": doc_seg['pdf_bytes'],
                "alertas": alertas_segmento if alertas_segmento else None
            })
        
        return {
            "documentos_finales": documentos_finales,
            "clasificaciones": clasificaciones,
            "error": None
        }
        
    except Exception as e:
        return {
            "documentos_finales": [],
            "clasificaciones": [],
            "error": str(e)
        }


@sgd_router.get("/consultar/{codigo_despacho}")
async def consultar_despacho(codigo_despacho: str, token: str = Depends(verify_api_token)):
    """
    Consulta la información del despacho y lista los documentos disponibles.
    Soporta tanto ID interno como código visible.
    """
    if not settings.BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="BEARER_TOKEN no configurado")
    
    datos_despacho_detalle = await consultar_despacho_detalle(codigo_despacho, settings.BEARER_TOKEN)
    
    if datos_despacho_detalle and "data" in datos_despacho_detalle:
        despacho_data = datos_despacho_detalle["data"]
        
        if isinstance(despacho_data, dict):
            codigo_visible = str(despacho_data.get("codigo", "N/A"))
            cliente = despacho_data.get("cliente", {})
            cliente_nombre = cliente.get("nombre", "N/A") if isinstance(cliente, dict) else "N/A"
            
            documentos_list = despacho_data.get("documentos", [])
            if not isinstance(documentos_list, list):
                documentos_list = []
            
            documentos_simplificados = []
            for doc in documentos_list:
                if isinstance(doc, dict):
                    tipo = doc.get("tipo", {})
                    documentos_simplificados.append({
                        "nombre": tipo.get("nombre", "Sin nombre") if isinstance(tipo, dict) else "Sin nombre",
                        "estado": doc.get("estado", "N/A"),
                        "fecha_recepcion": doc.get("fecha_recepcion", "N/A")
                    })
            
            usuarios_list = despacho_data.get("usuarios", [])
            if not isinstance(usuarios_list, list):
                usuarios_list = []
            
            pedidores = []
            jefes_operaciones = []
            
            for user in usuarios_list:
                if isinstance(user, dict):
                    role_name = user.get("role_name", "")
                    nombre = user.get("name", "")
                    
                    if role_name in ("pedidor", "pedidor_exportaciones"):
                        pedidores.append(nombre)
                    elif role_name == "jefe_operaciones":
                        jefes_operaciones.append(nombre)
            
            usuarios_datos = {
                "pedidor": pedidores if pedidores else None,
                "jefe_operaciones": jefes_operaciones if jefes_operaciones else None
            }
            
            return {
                "codigo_despacho": codigo_visible,
                "id_interno": str(despacho_data.get("id", "N/A")),
                "cliente": cliente_nombre,
                "estado": despacho_data.get("estado_despacho", "N/A"),
                "tipo": despacho_data.get("tipo_despacho", "N/A"),
                "total_documentos": len(documentos_simplificados),
                "documentos": documentos_simplificados,
                "usuarios": usuarios_datos
            }
    
    documentos_base64_list = await consultar_documentacion(codigo_despacho, settings.BEARER_TOKEN)
    
    if documentos_base64_list and isinstance(documentos_base64_list, list):
        documentos_info = []
        for doc in documentos_base64_list:
            if isinstance(doc, dict):
                documentos_info.append({
                    "nombre": doc.get("nombre_documento", "Sin nombre"),
                    "documento_id": doc.get("documento_id", "N/A")
                })
        
        return {
            "codigo": codigo_despacho,
            "mensaje": "Información limitada: solo se encontraron documentos",
            "total_documentos": len(documentos_info),
            "documentos": documentos_info
        }
    
    raise HTTPException(status_code=404, detail="Despacho no encontrado")


@sgd_router.post("/procesar/{codigo_despacho}", response_model=ProcesamientoResponse)
async def procesar_despacho(codigo_despacho: str, token: str = Depends(verify_api_token)):
    """
    Procesa el despacho completo:
    1. Análisis y corrección de calidad
    2. Clasificación de páginas
    3. Segmentación de documentos
    
    Almacena solo documentos finales segmentados en memoria.
    """
    if not settings.BEARER_TOKEN:
        raise HTTPException(status_code=500, detail="BEARER_TOKEN no configurado")
    
    datos_despacho_detalle = await consultar_despacho_detalle(codigo_despacho, settings.BEARER_TOKEN)
    
    codigo_visible = None
    cliente_nombre = "N/A"
    estado_despacho = "N/A"
    tipo_despacho = "N/A"
    
    if datos_despacho_detalle and "data" in datos_despacho_detalle:
        despacho_data = datos_despacho_detalle["data"]
        if isinstance(despacho_data, dict):
            codigo_visible = str(despacho_data.get("codigo", ""))
            cliente = despacho_data.get("cliente", {})
            cliente_nombre = cliente.get("nombre", "N/A") if isinstance(cliente, dict) else "N/A"
            estado_despacho = despacho_data.get("estado_despacho", "N/A")
            tipo_despacho = despacho_data.get("tipo_despacho", "N/A")
    
    documentos_base64_list = None
    
    if codigo_visible:
        documentos_base64_list = await consultar_documentacion(codigo_visible, settings.BEARER_TOKEN)
    
    if not documentos_base64_list:
        documentos_base64_list = await consultar_documentacion(codigo_despacho, settings.BEARER_TOKEN)
    
    if not documentos_base64_list or not isinstance(documentos_base64_list, list):
        raise HTTPException(status_code=404, detail="No se pudieron obtener los documentos")
    
    todos_documentos_finales = []
    
    for doc in documentos_base64_list:
        if isinstance(doc, dict):
            base64_data = doc.get("documento", "")
            
            if ',' in base64_data:
                _, base64_content = base64_data.split(',', 1)
            else:
                base64_content = base64_data
            
            try:
                file_bytes = base64.b64decode(base64_content)
                nombre_documento = doc.get("nombre_documento", "documento.pdf")
                
                validar_tamano_archivo(file_bytes)
                
                if es_archivo_excel(nombre_documento):
                    if not validar_excel(file_bytes, nombre_documento):
                        continue
                    try:
                        pdf_bytes = await convertir_excel_a_pdf(file_bytes, nombre_documento)
                        nombre_procesamiento = nombre_documento.rsplit('.', 1)[0] + '.pdf'
                    except Exception:
                        continue
                else:
                    if not validar_pdf(file_bytes):
                        continue
                    pdf_bytes = file_bytes
                    nombre_procesamiento = nombre_documento
                
                resultado = await procesar_pdf_completo(pdf_bytes, nombre_procesamiento)
                
                if resultado["error"]:
                    continue
                
                todos_documentos_finales.extend(resultado["documentos_finales"])
                
            except Exception:
                continue
    
    documentos_finales_sgd[codigo_despacho] = todos_documentos_finales
    
    docs_response = []
    for doc_final in todos_documentos_finales:
        docs_response.append(DocumentoFinal(
            archivo_origen=doc_final["archivo_origen"],
            nombre_salida=doc_final["nombre_salida"],
            tipo=doc_final["tipo"],
            paginas=doc_final["paginas"],
            alertas=[Alerta(**alerta) for alerta in doc_final["alertas"]] if doc_final["alertas"] else None
        ))
    
    return ProcesamientoResponse(
        codigo_despacho=codigo_despacho,
        cliente=cliente_nombre,
        estado=estado_despacho,
        tipo=tipo_despacho,
        total_documentos_segmentados=len(docs_response),
        documentos=docs_response
    )


@documentos_router.post("/procesar", response_model=ProcesamientoIndividualResponse)
async def procesar_documento_individual(file: UploadFile = File(...), token: str = Depends(verify_api_token)):
    """
    Procesa un documento individual completo:
    - Soporta archivos PDF y Excel (.xls, .xlsx, .xlsm, etc.)
    - Los archivos Excel se convierten automáticamente a PDF
    - Valida el tipo MIME real del archivo
    
    Flujo:
    1. Validación de tipo de archivo
    2. Conversión Excel a PDF (si aplica)
    3. Análisis y corrección de calidad
    4. Clasificación de páginas
    5. Segmentación de documentos
    
    Almacena solo documentos finales segmentados en memoria.
    """
    nombre_archivo = file.filename.lower()
    es_pdf = nombre_archivo.endswith('.pdf')
    es_excel = es_archivo_excel(file.filename)
    
    if not es_pdf and not es_excel:
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos PDF o Excel (.xls, .xlsx, .xlsm, .xlsb, .xltx, .xltm)"
        )
    
    try:
        file_bytes = await file.read()
        
        validar_tamano_archivo(file_bytes)
        
        if es_excel:
            if not validar_excel(file_bytes, file.filename):
                raise HTTPException(
                    status_code=400,
                    detail="El archivo no es un Excel válido"
                )
            pdf_bytes = await convertir_excel_a_pdf(file_bytes, file.filename)
            nombre_procesamiento = file.filename.rsplit('.', 1)[0] + '.pdf'
        else:
            if not validar_pdf(file_bytes):
                raise HTTPException(
                    status_code=400,
                    detail="El archivo no es un PDF válido"
                )
            pdf_bytes = file_bytes
            nombre_procesamiento = file.filename
        
        resultado = await procesar_pdf_completo(pdf_bytes, nombre_procesamiento)
        
        if resultado["error"]:
            raise HTTPException(status_code=500, detail=f"Error al procesar: {resultado['error']}")
        
        documentos_finales_individuales[file.filename] = resultado["documentos_finales"]
        
        docs_response = []
        for doc_final in resultado["documentos_finales"]:
            docs_response.append(DocumentoFinal(
                archivo_origen=doc_final["archivo_origen"],
                nombre_salida=doc_final["nombre_salida"],
                tipo=doc_final["tipo"],
                paginas=doc_final["paginas"],
                alertas=[Alerta(**alerta) for alerta in doc_final["alertas"]] if doc_final["alertas"] else None
            ))
        
        return ProcesamientoIndividualResponse(
            archivo_origen=file.filename,
            total_documentos_segmentados=len(docs_response),
            documentos=docs_response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar documento: {str(e)}")


# ============================================
# ADMIN ENDPOINTS - Gestión de Tokens
# ============================================

@admin_router.get("/", response_model=List[TokenInfo])
async def listar_tokens(admin_token: str = Depends(verify_admin_token)):
    """
    Lista todos los tokens de la API con su metadata.

    **Requiere token de administrador.**

    - Muestra tokens enmascarados por seguridad
    - Incluye fecha de creación y último uso
    - Muestra estado activo/inactivo
    """
    tokens = token_manager.list_tokens()
    return tokens


@admin_router.post("/generate", response_model=TokenCreateResponse)
async def generar_token(
    request: TokenCreateRequest,
    admin_token: str = Depends(verify_admin_token)
):
    """
    Genera un nuevo token de autenticación API.

    **Requiere token de administrador.**

    - **name**: Nombre descriptivo del token o usuario (ej: "Token para Postman", "Usuario Pablo", "App Móvil")

    **IMPORTANTE**: El token completo solo se muestra una vez.
    Guárdalo en un lugar seguro.
    """
    result = token_manager.generate_token(
        name=request.name,
        created_by="admin"
    )
    return result


@admin_router.delete("/{token_id}", response_model=TokenDeleteResponse)
async def eliminar_token(
    token_id: str,
    admin_token: str = Depends(verify_admin_token)
):
    """
    Elimina un token por su ID.

    **Requiere token de administrador.**

    - **token_id**: ID del token a eliminar (obtenido desde listar_tokens)

    El token eliminado dejará de funcionar inmediatamente.
    """
    success = token_manager.delete_token(token_id)

    if success:
        return TokenDeleteResponse(
            success=True,
            message=f"Token {token_id} eliminado exitosamente"
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Token {token_id} no encontrado"
        )


app.include_router(sgd_router)
app.include_router(documentos_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    return {"status": "online", "service": "API Docs"}