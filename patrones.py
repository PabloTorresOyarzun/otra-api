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
        "CONOCIMIENTO DE EMBARQUE", "GUÍA AÉREA", "DETALLE DE LA AEROGUIA", "CARTA DE PORTE",
        # Inglés
        "BILL OF LADING", "B/L", "AIR WAYBILL", "ROAD WAYBILL", "SEA WAYBILL",
        "COMBINED TRANSPORT", "PORT TO PORT SHIPMENT",
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
        "CERTIFICADO DE ORIGEN", "CERTIFICACION DE ORIGEN",
        # Inglés
        "CERTIFICATE OF ORIGIN", "CERTIFICATION OF ORIGIN",
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
    
    # 5. CERTIFICADO SANITARIO (Productos para consumo humano, farmacéuticos)
    "CERTIFICADO_SANITARIO": [
        # Español
        "CERTIFICADO SANITARIO", "CERTIFICADO DE SALUD", "CERTIFICADO SANITARIO DE EXPORTACIÓN",
        # Inglés
        "HEALTH CERTIFICATE", "SANITARY CERTIFICATE", "PUBLIC HEALTH CERTIFICATE",
        # Alemán
        "GESUNDHEITSZEUGNIS", "GESUNDHEITSBESCHEINIGUNG",
        # Portugués
        "CERTIFICADO SANITÁRIO", "CERTIFICADO DE SAÚDE",
        # Neerlandés
        "GEZONDHEIDSCERTIFICAAT", "SANITAIR CERTIFICAAT",
        # Francés
        "CERTIFICAT SANITAIRE", "CERTIFICAT DE SANTÉ"
    ],
    
    # 6. CERTIFICADO FITOSANITARIO (Productos vegetales, controlado por SAG)
    "CERTIFICADO_FITOSANITARIO": [
        # Español
        "CERTIFICADO FITOSANITARIO", "CERTIFICADO FITOSANITARIO DE EXPORTACIÓN",
        "PERMISO FITOSANITARIO",
        # Inglés
        "PHYTOSANITARY CERTIFICATE", "PLANT HEALTH CERTIFICATE",
        # Alemán
        "PFLANZENGESUNDHEITSZEUGNIS", "PHYTOSANITÄRES ZERTIFIKAT",
        # Portugués
        "CERTIFICADO FITOSSANITÁRIO", "CERTIFICADO DE SANIDADE VEGETAL",
        # Neerlandés
        "FYTOSANITAIR CERTIFICAAT", "PLANTGEZONDHEID CERTIFICAAT",
        # Francés
        "CERTIFICAT PHYTOSANITAIRE", "CERTIFICAT DE SANTÉ VÉGÉTALE"
    ],
    
    # 7. CERTIFICADO VETERINARIO (Productos animales, controlado por SAG)
    "CERTIFICADO_VETERINARIO": [
        # Español
        "CERTIFICADO VETERINARIO", "CERTIFICADO ZOOSANITARIO", 
        "CERTIFICADO SANITARIO ANIMAL", "CERTIFICADO VETERINARIO DE EXPORTACIÓN",
        # Inglés
        "VETERINARY CERTIFICATE", "ANIMAL HEALTH CERTIFICATE", "ZOOSANITARY CERTIFICATE",
        # Alemán
        "VETERINÄRBESCHEINIGUNG", "TIERÄRZTLICHES ZEUGNIS",
        # Portugués
        "CERTIFICADO VETERINÁRIO", "CERTIFICADO ZOOSSANITÁRIO",
        # Neerlandés
        "VETERINAIR CERTIFICAAT", "DIERENGEZONDHEID CERTIFICAAT",
        # Francés
        "CERTIFICAT VÉTÉRINAIRE", "CERTIFICAT DE SANTÉ ANIMALE"
    ],
    
    # 8. CERTIFICADO DE ANALISIS (Especificaciones técnicas/químicas industriales)
    "CERTIFICADO_ANALISIS": [
        # Español
        "CERTIFICADO DE ANÁLISIS", "CERTIFICADO DE ANALISIS", "HOJA DE ANÁLISIS",
        "INFORME DE ANÁLISIS",
        # Inglés
        "CERTIFICATE OF ANALYSIS", "ANALYTICAL CERTIFICATE", "TEST REPORT",
        "ANALYSIS REPORT", "COA",
        # Alemán
        "ANALYSEZERTIFIKAT", "PRÜFBERICHT", "ANALYSENBESCHEINIGUNG",
        # Portugués
        "CERTIFICADO DE ANÁLISE", "RELATÓRIO DE ANÁLISE",
        # Neerlandés
        "CERTIFICAAT VAN ANALYSE", "ANALYSERAPPORT",
        # Francés
        "CERTIFICAT D'ANALYSE", "RAPPORT D'ANALYSE"
    ],
    
    # 9. PÓLIZA DE SEGURO (DOCUMENTO DE VALOR)
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
    ],

    # 10. DECLARACION DE INGRESO
    "DECLARACION DE INGRESO": [
        # Español
        "DECLARACIÓN DE INGRESO", "DECLARACION DE INGRESO",
        # Inglés
        "DECLARATION OF INCOME",
        # Alemán
        "EINKOMMENSERKLÄRUNG",
        # Portugués
        "DECLARAÇÃO DE RENDA",
        # Neerlandés
        "INKOMSTENVERKLARING",
        # Francés
        "DÉCLARATION DE REVENU"
    ],

    # 11. DECLARACION JURADA
    "DECLARACION JURADA": [
        # Español
        "DECLARACIÓN JURADA",
        # Inglés
        "SWORN STATEMENT",
        # Alemán
        "EIDESSTATTLICHE ERKLÄRUNG",
        # Portugués
        "DECLARAÇÃO JURAMENTADA",
        # Neerlandés
        "GEZWOREN VERKLARING",
        # Francés
        "DÉCLARATION SOUS SERMENT"
    ],

    # 12. CARTA DE REMISION (Lista de documentos enviados)
    "CARTA_REMISION": [
        # Español
        "CARTA DE REMISIÓN", "CARTA REMISORIA", "CARTA DE ENVÍO",
        # Inglés
        "TRANSMITTAL LETTER", "TRANSMITTAL", "DOCUMENT TRANSMITTAL", "COVERING LETTER",
        # Alemán
        "BEGLEITSCHREIBEN", "SENDESCHREIBEN",
        # Portugués
        "CARTA DE REMESSA", "CARTA DE TRANSMISSÃO",
        # Neerlandés
        "BEGELEIDENDE BRIEF", "VERZENDBRIEF",
        # Francés
        "LETTRE DE TRANSMISSION", "LETTRE D'ENVOI"
    ],

    # 13. DOCUMENTO DE MENSAJERIA (Envío de documentos originales vía courier)
    "DOCUMENTO_MENSAJERIA": [
        # Términos generales en múltiples idiomas
        # Español
        "GUÍA DE MENSAJERÍA", "ENVÍO DE DOCUMENTOS", "COURIER",
        # Inglés
        "COURIER WAYBILL", "DOCUMENT SHIPMENT", "EXPRESS SHIPMENT",
        # Alemán
        "KURIERSENDUNG", "DOKUMENTENVERSAND",
        # Portugués
        "GUIA DE COURIER", "ENVIO DE DOCUMENTOS",
        # Neerlandés
        "KOERIERSZENDING", "DOCUMENTENVERZENDING",
        # Francés
        "ENVOI PAR COURSIER", "EXPÉDITION DE DOCUMENTS",
        
        # Servicios específicos DHL
        "DOX", "EXPRESS WORLDWIDE", "DHL EXPRESS", "WAYBILL DOC",
        
        # Servicios específicos FedEx
        "FEDEX ENVELOPE", "PRIORITY OVERNIGHT", "INTERNATIONAL DOCUMENT",
        "FEDEX INTERNATIONAL PRIORITY",
        
        # Servicios específicos UPS
        "UPS EXPRESS ENVELOPE", "UPS WORLDWIDE EXPRESS", "UPS DOCUMENT",
        
        # Servicios específicos TNT
        "TNT EXPRESS DOCUMENT", "TNT ENVELOPE",
        
        # Servicios específicos Aramex
        "ARAMEX DOCUMENT", "ARAMEX EXPRESS"
    ],

    # 14. AVISO DE RETENCION (Notificación de retención en aduana/courier)
    "AVISO_RETENCION": [
        # Español
        "AVISO DE RETENCIÓN", "AVISO DE RETENCION", "NOTIFICACIÓN DE RETENCIÓN",
        "NOTIFICACION DE RETENCION",
        # Inglés
        "RETENTION NOTICE", "HOLD NOTICE", "CUSTOMS HOLD", "SHIPMENT ON HOLD",
        "DETENTION NOTICE", "CARGO HOLD NOTICE",
        # Alemán
        "ZURÜCKHALTUNGSMITTEILUNG", "ZOLLVORMERKUNG", "SENDUNG ZURÜCKGEHALTEN",
        # Portugués
        "AVISO DE RETENÇÃO", "NOTIFICAÇÃO DE RETENÇÃO", "RETENÇÃO ADUANEIRA",
        # Neerlandés
        "KENNISGEVING VAN BEWARING", "DOUANE HOLD", "ZENDING VASTGEHOUDEN",
        # Francés
        "AVIS DE RÉTENTION", "AVIS DE RETENUE", "RÉTENTION DOUANIÈRE"
    ]
}

# El patrón predeterminado (si no se encuentra ningún patrón en el documento)
PATRON_DEFAULT = "UNKNOWN_DOCUMENT"