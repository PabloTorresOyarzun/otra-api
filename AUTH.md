# Autenticación API

Este documento explica cómo configurar y usar la autenticación con tokens para la API REST.

## Configuración del Servidor

### 1. Configurar tokens válidos

Edita tu archivo `.env` y agrega los tokens que quieres permitir:

```env
API_TOKENS=token1,token2,token3
```

Puedes usar cualquier cantidad de tokens, separados por comas.

**Recomendación para generar tokens seguros:**

```bash
# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# OpenSSL
openssl rand -base64 32

# Node.js
node -e "console.log(require('crypto').randomBytes(32).toString('base64url'))"
```

### 2. Ejemplo de configuración

```env
BEARER_TOKEN=tu-bearer-token-sgd
AZURE_ENDPOINT=https://tu-endpoint.cognitiveservices.azure.com/
AZURE_KEY=tu-azure-key
API_TOKENS=AbCdEf123456XyZ,OtroTokenSeguro789,token-para-testing
```

## Uso desde Postman

### Opción 1: Usando Authorization Tab (Recomendado)

1. Abre tu request en Postman
2. Ve a la pestaña **Authorization**
3. En **Type**, selecciona **Bearer Token**
4. En **Token**, pega uno de tus tokens configurados
5. Haz clic en **Send**

![Postman Authorization](https://i.imgur.com/example.png)

### Opción 2: Header Manual

1. Ve a la pestaña **Headers**
2. Agrega un nuevo header:
   - **Key**: `Authorization`
   - **Value**: `Bearer tu-token-aqui`
3. Haz clic en **Send**

### Ejemplos de Requests

#### 1. Consultar Despacho

```
GET http://localhost:8000/sgd/consultar/123456
Authorization: Bearer AbCdEf123456XyZ
```

#### 2. Procesar Despacho

```
POST http://localhost:8000/sgd/procesar/123456
Authorization: Bearer AbCdEf123456XyZ
```

#### 3. Procesar Documento Individual

```
POST http://localhost:8000/documentos/procesar
Authorization: Bearer AbCdEf123456XyZ
Content-Type: multipart/form-data

Body (form-data):
- file: [seleccionar archivo PDF o Excel]
```

## Uso desde cURL

```bash
# Consultar despacho
curl -X GET "http://localhost:8000/sgd/consultar/123456" \
  -H "Authorization: Bearer AbCdEf123456XyZ"

# Procesar despacho
curl -X POST "http://localhost:8000/sgd/procesar/123456" \
  -H "Authorization: Bearer AbCdEf123456XyZ"

# Procesar documento individual
curl -X POST "http://localhost:8000/documentos/procesar" \
  -H "Authorization: Bearer AbCdEf123456XyZ" \
  -F "file=@/ruta/al/documento.pdf"
```

## Uso desde Python

```python
import requests

# Token de autorización
TOKEN = "AbCdEf123456XyZ"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Consultar despacho
response = requests.get(
    "http://localhost:8000/sgd/consultar/123456",
    headers=headers
)
print(response.json())

# Procesar documento
with open("documento.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/documentos/procesar",
        headers=headers,
        files=files
    )
print(response.json())
```

## Uso desde JavaScript/Fetch

```javascript
const TOKEN = "AbCdEf123456XyZ";

// Consultar despacho
fetch("http://localhost:8000/sgd/consultar/123456", {
  headers: {
    "Authorization": `Bearer ${TOKEN}`
  }
})
  .then(res => res.json())
  .then(data => console.log(data));

// Procesar documento
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:8000/documentos/procesar", {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${TOKEN}`
  },
  body: formData
})
  .then(res => res.json())
  .then(data => console.log(data));
```

## Códigos de Respuesta

### Éxito
- **200 OK**: Request exitoso
- **201 Created**: Recurso creado exitosamente

### Errores de Autenticación
- **401 Unauthorized**: Token inválido o no proporcionado
  ```json
  {
    "detail": "Token de autenticación inválido"
  }
  ```

- **500 Internal Server Error**: API_TOKENS no configurado en el servidor
  ```json
  {
    "detail": "API_TOKENS no configurado en el servidor"
  }
  ```

### Otros Errores
- **400 Bad Request**: Datos inválidos
- **404 Not Found**: Recurso no encontrado
- **413 Payload Too Large**: Archivo excede tamaño máximo
- **408 Request Timeout**: Procesamiento excedió tiempo límite

## Documentación Interactiva

Una vez que la API esté corriendo, puedes acceder a la documentación interactiva:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

En Swagger UI puedes:
1. Hacer clic en el botón **Authorize** (🔒)
2. Ingresar tu token (sin el prefijo "Bearer")
3. Hacer clic en **Authorize**
4. Probar todos los endpoints directamente desde el navegador

## Seguridad

### Mejores Prácticas

1. **Nunca compartas tus tokens** en repositorios públicos
2. **Usa tokens diferentes** para cada ambiente (desarrollo, staging, producción)
3. **Rota los tokens** periódicamente
4. **Usa HTTPS** en producción para evitar que los tokens sean interceptados
5. **Mantén el archivo `.env` en `.gitignore`**

### Rotación de Tokens

Para cambiar los tokens:

1. Genera nuevos tokens seguros
2. Actualiza el archivo `.env` con los nuevos tokens
3. Reinicia el servidor
4. Notifica a los usuarios para que actualicen sus configuraciones

## Troubleshooting

### Error: "Not authenticated"

Verifica que:
- Estés enviando el header `Authorization`
- El formato sea: `Bearer <token>`
- El token esté en la lista de `API_TOKENS` en `.env`
- No haya espacios extra en el token

### Error: "API_TOKENS no configurado"

- Asegúrate de que el archivo `.env` exista
- Verifica que la variable `API_TOKENS` esté definida
- Reinicia el servidor después de modificar `.env`

### El token no funciona después de agregarlo

- Reinicia el servidor API (las variables de entorno se cargan al inicio)
- Verifica que no haya espacios extra en el `.env`
- Confirma que el token en el cliente sea exactamente igual al del servidor
