import httpx
import asyncio
import base64
import zipfile
import rarfile
import io
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from config import get_settings


settings = get_settings()
security = HTTPBearer()
modelos_router = APIRouter(prefix="/modelos", tags=["Modelos Azure DI"])


def verify_admin_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    Verifica que el token de administrador sea válido.
    Solo permite acceso a endpoints de gestión de modelos.

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


AZURE_DI_ENDPOINT = "http://azure-di-custom:5000"
API_VERSION = "2022-08-31"

# Modelos esperados del sistema
MODELOS_ESPERADOS = [
    {"id": "invoice", "nombre": "Factura Comercial", "descripcion": "Modelo para facturas comerciales"},
    {"id": "transport", "nombre": "Documento de Transporte", "descripcion": "Modelo para documentos de transporte (B/L, AWB, etc)"},
    {"id": "origin", "nombre": "Certificado de Origen", "descripcion": "Modelo para certificados de origen"},
    {"id": "packing-list", "nombre": "Lista de Embalaje", "descripcion": "Modelo para listas de embalaje/empaque"},
    {"id": "health", "nombre": "Certificado Sanitario", "descripcion": "Modelo para certificados sanitarios/fitosanitarios"},
    {"id": "insurance", "nombre": "Póliza de Seguro", "descripcion": "Modelo para pólizas y certificados de seguro"}
]


class ModeloDisponibilidad(BaseModel):
    id: str
    nombre: str
    descripcion: str
    entrenado: bool
    fecha_creacion: Optional[str] = None


class ListarModelosResponse(BaseModel):
    total_esperados: int
    total_entrenados: int
    modelos: List[ModeloDisponibilidad]


class EntrenarResponse(BaseModel):
    success: bool
    model_id: str
    message: str


async def obtener_modelos_entrenados() -> Dict[str, Dict]:
    """Obtiene todos los modelos custom entrenados del contenedor"""
    url = f"{AZURE_DI_ENDPOINT}/formrecognizer/documentModels?api-version={API_VERSION}"
    
    timeout_config = httpx.Timeout(
        connect=settings.TIMEOUT_CONNECT,
        read=settings.TIMEOUT_READ,
        write=settings.TIMEOUT_WRITE,
        pool=5.0
    )
    
    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.get(url)
            
            if response.status_code == 500:
                return {}
            
            if response.status_code == 200:
                data = response.json()
                modelos = data.get('value', [])
                
                # Crear diccionario indexado por modelId
                modelos_dict = {}
                for modelo in modelos:
                    model_id = modelo.get('modelId', '')
                    # Solo incluir modelos custom (los que están en MODELOS_ESPERADOS)
                    if any(m['id'] == model_id for m in MODELOS_ESPERADOS):
                        modelos_dict[model_id] = {
                            'createdDateTime': modelo.get('createdDateTime'),
                            'description': modelo.get('description')
                        }
                
                return modelos_dict
            
            return {}
    except:
        return {}


def extraer_archivos_comprimido(file_bytes: bytes, filename: str) -> bytes:
    """Extrae archivos de ZIP o RAR y crea un nuevo ZIP en memoria"""
    zip_buffer = io.BytesIO()
    
    try:
        # Intentar como ZIP
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                    for name in zip_ref.namelist():
                        data = zip_ref.read(name)
                        new_zip.writestr(name, data)
        
        # Intentar como RAR
        elif filename.lower().endswith('.rar'):
            with rarfile.RarFile(io.BytesIO(file_bytes), 'r') as rar_ref:
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                    for name in rar_ref.namelist():
                        data = rar_ref.read(name)
                        new_zip.writestr(name, data)
        else:
            raise ValueError("Formato no soportado. Use ZIP o RAR")
        
        zip_buffer.seek(0)
        return base64.b64encode(zip_buffer.read()).decode('utf-8')
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error al procesar archivo comprimido: {str(e)}"
        )


async def verificar_estado_entrenamiento(operation_location: str, max_intentos: int = 120) -> Dict:
    """Verifica el estado del entrenamiento con polling"""
    timeout_config = httpx.Timeout(
        connect=settings.TIMEOUT_CONNECT,
        read=settings.TIMEOUT_READ,
        write=settings.TIMEOUT_WRITE,
        pool=5.0
    )
    
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        for _ in range(max_intentos):
            response = await client.get(operation_location)
            data = response.json()
            
            status = data.get('status')
            
            if status == 'succeeded':
                return {
                    "status": "succeeded",
                    "result": data.get('result', {})
                }
            elif status == 'failed':
                return {
                    "status": "failed",
                    "error": data.get('error', {})
                }
            
            await asyncio.sleep(5)
    
    return {
        "status": "timeout",
        "error": "El entrenamiento excedió el tiempo máximo de espera"
    }


@modelos_router.get("/listar", response_model=ListarModelosResponse)
async def listar_modelos(admin_token: str = Depends(verify_admin_token)):
    """
    Lista todos los modelos del sistema.
    
    **Requiere token de administrador.**
    
    Muestra:
    - Modelos esperados del sistema (invoice, transport, origin, etc.)
    - Estado de entrenamiento (entrenado o no entrenado)
    - Fecha de creación si está entrenado
    """
    modelos_entrenados = await obtener_modelos_entrenados()
    
    modelos_lista = []
    for modelo_esperado in MODELOS_ESPERADOS:
        model_id = modelo_esperado['id']
        esta_entrenado = model_id in modelos_entrenados
        
        modelos_lista.append(ModeloDisponibilidad(
            id=model_id,
            nombre=modelo_esperado['nombre'],
            descripcion=modelo_esperado['descripcion'],
            entrenado=esta_entrenado,
            fecha_creacion=modelos_entrenados.get(model_id, {}).get('createdDateTime') if esta_entrenado else None
        ))
    
    return ListarModelosResponse(
        total_esperados=len(MODELOS_ESPERADOS),
        total_entrenados=len(modelos_entrenados),
        modelos=modelos_lista
    )


@modelos_router.post("/entrenar", response_model=EntrenarResponse)
async def entrenar_modelo(
    model_id: str = Form(...),
    archivo: UploadFile = File(...),
    admin_token: str = Depends(verify_admin_token)
):
    """
    Entrena un modelo custom.
    
    **Requiere token de administrador.**
    
    Args:
        model_id: ID del modelo a entrenar (invoice, transport, origin, packing-list, health, insurance)
        archivo: Archivo ZIP o RAR con los PDFs y archivos .labels.json
    
    El archivo debe contener:
    - PDFs de ejemplo
    - Archivos .labels.json correspondientes a cada PDF
    
    Ejemplo de contenido del ZIP:
    - doc1.pdf
    - doc1.pdf.labels.json
    - doc2.pdf
    - doc2.pdf.labels.json
    """
    # Validar que el model_id sea uno de los esperados
    if not any(m['id'] == model_id for m in MODELOS_ESPERADOS):
        modelos_validos = [m['id'] for m in MODELOS_ESPERADOS]
        raise HTTPException(
            status_code=400,
            detail=f"model_id inválido. Valores permitidos: {', '.join(modelos_validos)}"
        )
    
    # Validar extensión del archivo
    filename = archivo.filename.lower()
    if not (filename.endswith('.zip') or filename.endswith('.rar')):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos ZIP o RAR"
        )
    
    try:
        # Leer archivo
        file_bytes = await archivo.read()
        
        # Extraer y convertir a ZIP base64
        base64_zip = extraer_archivos_comprimido(file_bytes, archivo.filename)
        
        # Obtener descripción del modelo
        descripcion = next(
            (m['descripcion'] for m in MODELOS_ESPERADOS if m['id'] == model_id),
            f"Modelo {model_id}"
        )
        
        # Iniciar entrenamiento
        url = f"{AZURE_DI_ENDPOINT}/formrecognizer/documentModels:build?api-version={API_VERSION}"
        
        payload = {
            "modelId": model_id,
            "description": descripcion,
            "buildMode": "template",
            "base64Source": base64_zip
        }
        
        timeout_config = httpx.Timeout(
            connect=settings.TIMEOUT_CONNECT,
            read=600.0,
            write=settings.TIMEOUT_WRITE,
            pool=5.0
        )
        
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.post(url, json=payload)
            
            if response.status_code == 409:
                # El modelo ya existe, intentar eliminarlo primero
                delete_url = f"{AZURE_DI_ENDPOINT}/formrecognizer/documentModels/{model_id}?api-version={API_VERSION}"
                delete_response = await client.delete(delete_url)
                
                if delete_response.status_code in [204, 404]:
                    # Reintentar entrenamiento después de eliminar
                    response = await client.post(url, json=payload)
                    
                    if response.status_code != 202:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Error al reintentar entrenamiento: {response.text}"
                        )
                else:
                    raise HTTPException(
                        status_code=409,
                        detail=f"El modelo '{model_id}' ya existe y no se pudo eliminar. Elimínalo manualmente primero usando: DELETE /modelos/eliminar/{model_id}"
                    )
            elif response.status_code != 202:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error iniciando entrenamiento: {response.text}"
                )
            
            operation_location = response.headers.get('Operation-Location')
            
            if not operation_location:
                raise HTTPException(
                    status_code=500,
                    detail="No se recibió Operation-Location del servidor"
                )
            
            # Verificar estado del entrenamiento
            resultado = await verificar_estado_entrenamiento(operation_location)
            
            if resultado["status"] == "succeeded":
                return EntrenarResponse(
                    success=True,
                    model_id=model_id,
                    message=f"Modelo {model_id} entrenado exitosamente"
                )
            else:
                error_msg = resultado.get("error", {}).get("message", "Error desconocido")
                return EntrenarResponse(
                    success=False,
                    model_id=model_id,
                    message=f"Error al entrenar modelo: {error_msg}"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el entrenamiento: {str(e)}"
        )


@modelos_router.delete("/eliminar/{model_id}")
async def eliminar_modelo(model_id: str, admin_token: str = Depends(verify_admin_token)):
    """
    Elimina un modelo entrenado.
    
    **Requiere token de administrador.**
    
    Args:
        model_id: ID del modelo a eliminar (invoice, transport, origin, packing-list, health, insurance)
    """
    # Validar que el model_id sea uno de los esperados
    if not any(m['id'] == model_id for m in MODELOS_ESPERADOS):
        modelos_validos = [m['id'] for m in MODELOS_ESPERADOS]
        raise HTTPException(
            status_code=400,
            detail=f"model_id inválido. Valores permitidos: {', '.join(modelos_validos)}"
        )
    
    url = f"{AZURE_DI_ENDPOINT}/formrecognizer/documentModels/{model_id}?api-version={API_VERSION}"
    
    timeout_config = httpx.Timeout(
        connect=settings.TIMEOUT_CONNECT,
        read=settings.TIMEOUT_READ,
        write=settings.TIMEOUT_WRITE,
        pool=5.0
    )
    
    try:
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            response = await client.delete(url)
            
            if response.status_code == 204:
                return {
                    "success": True,
                    "message": f"Modelo {model_id} eliminado exitosamente"
                }
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Modelo {model_id} no encontrado o no está entrenado"
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error al eliminar modelo: {response.text}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar modelo: {str(e)}"
        )