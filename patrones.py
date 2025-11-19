# --- CONFIGURACION DE PATRONES DE INICIO (ADUANAS CHILE) ---

# Define los patrones de inicio de documento. 
# La clave es el nombre de la CLASIFICACIÓN (usando nombres estándar en español),
# y el valor es una LISTA de textos que indican el inicio de ese documento 
# en múltiples idiomas: Español, Inglés, Alemán, Portugués, Neerlandés y Francés.

PATRONES_INICIO = {
    
    # 1. DOCUMENTO PRINCIPAL DE VENTA: FACTURA COMERCIAL
    "FACTURA_COMERCIAL": [
        # Español
        "FACTURA COMERCIAL", "FACTURA", 
        # Inglés
        "COMMERCIAL INVOICE", "INVOICE", 
        # Alemán
        "HANDELSRECHNUNG", "RECHNUNG", 
        # Portugués
        "FATURA COMERCIAL", "FATURA", 
        # Neerlandés
        "HANDELSFACTUUR", "FACTUUR",
        # Francés
        "FACTURE COMMERCIALE", "FACTURE"
    ],
    
    # 2. DOCUMENTOS DE TRANSPORTE (Marítimo/Aéreo/Terrestre)
    "DOCUMENTO_TRANSPORTE": [
        # Español
        "CONOCIMIENTO DE EMBARQUE", "GUÍA AÉREA", "CARTA DE PORTE",
        # Inglés
        "BILL OF LADING", "B/L", "WAYBILL", "AIR WAYBILL", "ROAD WAYBILL", "SEA WAYBILL",
        # Alemán
        "FRACHTBRIEF", "LUFTFRACHTBRIEF", 
        # Portugués
        "CONHECIMENTO DE EMBARQUE", "CARTA DE PORTE", 
        # Neerlandés
        "ZEEVRACHTBRIEF", "VRACHTBRIEF",
        # Francés
        "CONNAISSEMENT", "LETTRE DE TRANSPORT AÉRIEN", "LETTRE DE VOITURE"
    ],
    
    # 3. DOCUMENTOS DE CERTIFICACIÓN DE ORIGEN
    "CERTIFICADO_ORIGEN": [
        # Español
        "CERTIFICADO DE ORIGEN",
        # Inglés
        "CERTIFICATE OF ORIGIN", 
        # Alemán
        "URSPRUNGSZEUGNIS",
        # Portugués
        "CERTIFICADO DE ORIGEM", 
        # Neerlandés
        "CERTIFICAAT VAN OORSPRONG",
        # Francés
        "CERTIFICAT D'ORIGINE"
    ],
    
    # 4. DOCUMENTOS DE DETALLE: LISTA DE EMBALAJE
    "LISTA_EMBALAJE": [
        # Español
        "LISTA DE EMBALAJE", "LISTA DE EMPAQUE",
        # Inglés
        "PACKING LIST", "PACKING LIST ORDER", 
        # Alemán
        "PACKLISTE",
        # Portugués
        "LISTA DE EMBALAGEM", 
        # Neerlandés
        "PAKLIJST",
        # Francés
        "LISTE DE COLISAGE"
    ],
    
    # 5. CERTIFICADOS VETERINARIOS/FITOSANITARIOS (Controlados por SAG en Chile)
    "CERTIFICADO_SANITARIO": [
        # Español
        "CERTIFICADO SANITARIO", "CERTIFICADO FITOSANITARIO", "CERTIFICADO DE ANÁLISIS", "CERTIFICADO ZOOSANITARIO", 
        # Inglés
        "HEALTH CERTIFICATE", "PHYTOSANITARY CERTIFICATE", "CERTIFICATE OF ANALYSIS", "VETERINARY CERTIFICATE",
        # Alemán
        "GESUNDHEITSZEUGNIS", "ANALYSEZERTIFIKAT",
        # Portugués
        "CERTIFICADO SANITÁRIO", "CERTIFICADO DE ANÁLISE", 
        # Neerlandés
        "GEZONDHEIDSCERTIFICAAT", "CERTIFICAAT VAN ANALYSE",
        # Francés
        "CERTIFICAT SANITAIRE", "CERTIFICAT D'ANALYSE"
    ],
    
    # 6. PÓLIZA DE SEGURO (DOCUMENTO DE VALOR)
    "POLIZA_SEGURO": [
        # Español
        "PÓLIZA DE SEGURO", "CERTIFICADO DE SEGURO",
        # Inglés
        "INSURANCE POLICY", "INSURANCE CERTIFICATE", "COVER NOTE",
        # Alemán
        "VERSICHERUNGSPOLICE", "VERSICHERUNGSZERTIFIKAT",
        # Portugués
        "APÓLICE DE SEGURO", "CERTIFICADO DE SEGURO",
        # Neerlandés
        "VERZEKERINGSBEWIJS", "POLIS",
        # Francés
        "POLICE D'ASSURANCE", "CERTIFICAT D'ASSURANCE"
    ]
}

# El patrón predeterminado (si no se encuentra ningún patrón en el documento)
PATRON_DEFAULT = "UNKNOWN_DOCUMENT"