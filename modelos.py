import httpx
import asyncio
import base64
import zipfile
import rarfile
import io
from typing import List, Dict, Optional, Annotated
from litestar import Router, get, post, delete, Request
from litestar.exceptions import HTTPException
from litestar.datastructures import UploadFile
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.status_codes import HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN, HTTP_404_NOT_FOUND, HTTP_409_CONFLICT, HTTP_500_INTERNAL_SERVER_ERROR
from pydantic import BaseModel

from config import get_settings


settings = get_settings()


def get_bearer_token(request: Request) -> str:
    """Extrae el token Bearer del header Authorization."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token de autenticación requerido")
    return auth_header[7:]


def verify_admin_token(request: Request) -> str:
    """Verifica que el token de administrador sea válido."""
    token = get_bearer_token(request)
    
    if not settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_TOKEN no configurado en el servidor"
        )

    if token != settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Se requiere token de administrador"
        )

    return token


AZURE_DI_ENDPOINT = "http://azure-di-custom:5000"
API_VERSION = "2022-08-31"

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


class EntrenarRequest(BaseModel):
    model_id: str


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
                
                modelos_dict = {}
                for modelo in modelos:
                    model_id = modelo.get('modelId', '')
                    if any(m['id'] == model_id for m in MODELOS_ESPERADOS):
                        modelos_dict[model_id] = {
                            'createdDateTime': modelo.get('createdDateTime'),
                            'description': modelo.get('description')
                        }
                
                return modelos_dict
            
            return {}
    except:
        return {}


def extraer_archivos_comprimido(file_bytes: bytes, filename: str) -> str:
    """Extrae archivos de ZIP o RAR y crea un nuevo ZIP en memoria"""
    zip_buffer = io.BytesIO()
    
    try:
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as zip_ref:
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as new_zip:
                    for name in zip_ref.namelist():
                        data = zip_ref.read(name)
                        new_zip.writestr(name, data)
        
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
            status_code=HTTP_400_BAD_REQUEST,
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


@get("/listar")
async def listar_modelos(request: Request) -> ListarModelosResponse:
    """Lista todos los modelos del sistema."""
    verify_admin_token(request)
    
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


@post("/entrenar")
async def entrenar_modelo(
    request: Request,
    data: Annotated[UploadFile, Body(media_type=RequestEncodingType.MULTI_PART)],
    model_id: str
) -> EntrenarResponse:
    """Entrena un modelo custom."""
    verify_admin_token(request)
    
    if not any(m['id'] == model_id for m in MODELOS_ESPERADOS):
        modelos_validos = [m['id'] for m in MODELOS_ESPERADOS]
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"model_id inválido. Valores permitidos: {', '.join(modelos_validos)}"
        )
    
    filename = data.filename.lower()
    if not (filename.endswith('.zip') or filename.endswith('.rar')):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Solo se aceptan archivos ZIP o RAR"
        )
    
    try:
        file_bytes = await data.read()
        
        base64_zip = extraer_archivos_comprimido(file_bytes, data.filename)
        
        descripcion = next(
            (m['descripcion'] for m in MODELOS_ESPERADOS if m['id'] == model_id),
            f"Modelo {model_id}"
        )
        
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
                delete_url = f"{AZURE_DI_ENDPOINT}/formrecognizer/documentModels/{model_id}?api-version={API_VERSION}"
                delete_response = await client.delete(delete_url)
                
                if delete_response.status_code in [204, 404]:
                    response = await client.post(url, json=payload)
                    
                    if response.status_code != 202:
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Error al reintentar entrenamiento: {response.text}"
                        )
                else:
                    raise HTTPException(
                        status_code=HTTP_409_CONFLICT,
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
                    status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No se recibió Operation-Location del servidor"
                )
            
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
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el entrenamiento: {str(e)}"
        )


@delete("/eliminar/{model_id:str}", status_code=200)
async def eliminar_modelo(request: Request, model_id: str) -> dict:
    """Elimina un modelo entrenado."""
    verify_admin_token(request)
    
    if not any(m['id'] == model_id for m in MODELOS_ESPERADOS):
        modelos_validos = [m['id'] for m in MODELOS_ESPERADOS]
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
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
                    status_code=HTTP_404_NOT_FOUND,
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
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al eliminar modelo: {str(e)}"
        )


modelos_router = Router(
    path="/modelos",
    route_handlers=[listar_modelos, entrenar_modelo, eliminar_modelo],
    tags=["Modelos Azure DI"]
)