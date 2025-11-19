"""
Gestor de tokens API con persistencia en archivo JSON.
Permite listar, crear y eliminar tokens de autenticación.
"""
import json
import os
import secrets
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


class TokenManager:
    """Gestiona tokens API con almacenamiento en archivo JSON."""

    def __init__(self, tokens_file: str = "tokens.json"):
        self.tokens_file = Path(tokens_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Crea el archivo de tokens si no existe."""
        if not self.tokens_file.exists():
            self._save_tokens({})

    def _load_tokens(self) -> Dict:
        """Carga los tokens desde el archivo JSON."""
        try:
            with open(self.tokens_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_tokens(self, tokens: Dict):
        """Guarda los tokens en el archivo JSON."""
        with open(self.tokens_file, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, indent=2, ensure_ascii=False)

    def list_tokens(self) -> List[Dict]:
        """
        Lista todos los tokens con su metadata (sin exponer el token completo).

        Returns:
            Lista de diccionarios con información de cada token.
        """
        tokens = self._load_tokens()
        result = []

        for token_value, metadata in tokens.items():
            # Ocultar parte del token por seguridad
            masked_token = f"{token_value[:8]}...{token_value[-4:]}" if len(token_value) > 12 else "***"

            result.append({
                "id": metadata.get("id"),
                "name": metadata.get("name"),
                "masked_token": masked_token,
                "created_at": metadata.get("created_at"),
                "created_by": metadata.get("created_by", "system"),
                "last_used": metadata.get("last_used"),
                "is_active": metadata.get("is_active", True)
            })

        return sorted(result, key=lambda x: x["created_at"], reverse=True)

    def generate_token(self, name: str, created_by: str = "admin", length: int = 32) -> Dict:
        """
        Genera un nuevo token seguro.

        Args:
            name: Nombre descriptivo del token
            created_by: Quién creó el token
            length: Longitud del token en bytes (default: 32)

        Returns:
            Diccionario con el token generado y su metadata
        """
        tokens = self._load_tokens()

        # Generar token seguro
        token_value = secrets.token_urlsafe(length)
        token_id = secrets.token_hex(8)

        # Crear metadata
        metadata = {
            "id": token_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": created_by,
            "last_used": None,
            "is_active": True
        }

        # Guardar token
        tokens[token_value] = metadata
        self._save_tokens(tokens)

        return {
            "id": token_id,
            "token": token_value,
            "name": name,
            "created_at": metadata["created_at"],
            "message": "Token generado exitosamente. Guárdalo en un lugar seguro, no podrás verlo de nuevo."
        }

    def delete_token(self, token_id: str) -> bool:
        """
        Elimina un token por su ID.

        Args:
            token_id: ID del token a eliminar

        Returns:
            True si se eliminó exitosamente, False si no se encontró
        """
        tokens = self._load_tokens()

        # Buscar token por ID
        token_to_delete = None
        for token_value, metadata in tokens.items():
            if metadata.get("id") == token_id:
                token_to_delete = token_value
                break

        if token_to_delete:
            del tokens[token_to_delete]
            self._save_tokens(tokens)
            return True

        return False

    def get_all_valid_tokens(self) -> set:
        """
        Obtiene todos los tokens activos para validación.

        Returns:
            Set con todos los tokens activos
        """
        tokens = self._load_tokens()
        return {
            token_value
            for token_value, metadata in tokens.items()
            if metadata.get("is_active", True)
        }

    def is_valid_token(self, token: str) -> bool:
        """
        Verifica si un token es válido y está activo.

        Args:
            token: Token a verificar

        Returns:
            True si el token es válido, False en caso contrario
        """
        tokens = self._load_tokens()
        metadata = tokens.get(token)

        if metadata and metadata.get("is_active", True):
            # Actualizar último uso
            metadata["last_used"] = datetime.utcnow().isoformat()
            tokens[token] = metadata
            self._save_tokens(tokens)
            return True

        return False

    def update_last_used(self, token: str):
        """
        Actualiza la fecha de último uso de un token.

        Args:
            token: Token a actualizar
        """
        tokens = self._load_tokens()

        if token in tokens:
            tokens[token]["last_used"] = datetime.utcnow().isoformat()
            self._save_tokens(tokens)

    def deactivate_token(self, token_id: str) -> bool:
        """
        Desactiva un token sin eliminarlo (para auditoría).

        Args:
            token_id: ID del token a desactivar

        Returns:
            True si se desactivó exitosamente, False si no se encontró
        """
        tokens = self._load_tokens()

        for token_value, metadata in tokens.items():
            if metadata.get("id") == token_id:
                metadata["is_active"] = False
                tokens[token_value] = metadata
                self._save_tokens(tokens)
                return True

        return False

    def get_token_by_id(self, token_id: str) -> Optional[Dict]:
        """
        Obtiene la información de un token por su ID (sin exponer el token).

        Args:
            token_id: ID del token

        Returns:
            Diccionario con la metadata del token o None si no existe
        """
        tokens = self._load_tokens()

        for token_value, metadata in tokens.items():
            if metadata.get("id") == token_id:
                masked_token = f"{token_value[:8]}...{token_value[-4:]}"
                return {
                    **metadata,
                    "masked_token": masked_token
                }

        return None


# Instancia global del gestor de tokens
token_manager = TokenManager()
