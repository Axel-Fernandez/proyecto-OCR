import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
from datetime import datetime
from dateutil.parser import parse
import os
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import concurrent.futures


@dataclass
class LayerResult:
    """Estructura de datos para almacenar resultados de cada capa"""
    layer_name: str
    confidence: float
    data: Dict[str, Any]
    raw_text: str = ""


class LayerProcessor:
    """Clase base para procesadores de capas"""
    def __init__(self, name: str):
        self.name = name

    def process(self, image: np.ndarray) -> LayerResult:
        raise NotImplementedError("Debe implementar el método process")

class HeaderLayer(LayerProcessor):
    """Procesa la parte superior del recibo para información del negocio"""
    def __init__(self):
        super().__init__("header")
        self.config = '--psm 6'  # Asume un bloque uniforme de texto
        
    def process(self, image: np.ndarray) -> LayerResult:
        # Tomar el tercio superior de la imagen
        height = image.shape[0]
        header_image = image[:height//3, :]
        
        # Mejorar el contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        header_image = clahe.apply(header_image)
        
        # Aplicar procesamiento específico para encabezados
        # Usar umbral de Otsu en lugar de umbral adaptativo
        _, processed = cv2.threshold(header_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Aplicar dilatación suave para mejorar la conectividad de los caracteres
        kernel = np.ones((2,1), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        # Obtener texto con configuración específica
        text = pytesseract.image_to_data(
            processed, 
            output_type=pytesseract.Output.DICT,
            config=self.config
        )
        
        # Extraer información del negocio
        business_info = self._extract_business_info(text)
        confidence = np.mean([float(conf) for conf in text['conf'] if conf != '-1'])
        
        return LayerResult(
            layer_name="header",
            confidence=confidence,
            data=business_info,
            raw_text=" ".join(word for word in text['text'] if word.strip())
        )

    def _extract_business_info(self, text_data: Dict) -> Dict[str, str]:
        lines = []
        current_line = []
        last_top = None
        line_height_threshold = 50  # Umbral para considerar nueva línea

        # Agrupar palabras en líneas basado en posición vertical
        for i, (word, top) in enumerate(zip(text_data['text'], text_data['top'])):
            if not word.strip():
                continue
                
            if last_top is None:
                current_line.append(word)
            elif abs(top - last_top) > line_height_threshold:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
                
            last_top = top
            
        if current_line:
            lines.append(" ".join(current_line))

        info = {
            'nombre_empresa': None,
            'factura': None,
            'direccion': None,
            'telefono': None,
            'transaction_number': None,
            'authorization_number': None,
            'total': None,
            'ruc': None,
            'payment_method': None
        }

        tax_patterns = [
            r'(?:RUC)[:]?\s*([\d]{13})',
            r'[0-9]{13}'  
        ]
        phone_patterns = [
            r'(?:TEL|PHONE|TEL[EÉ]FONO)[:]?\s*([\d\-\(\)\s]{8,})',
            r'(?:\+?[\d]{2})?\s*[\d]{2,3}[\s\-]?[\d]{3,4}[\s\-]?[\d]{4}'
        ]

        for line in lines:
            clean_line = line.strip()
            
            if not info['nombre_empresa'] and not any(char.isdigit() for char in clean_line):
                if len(clean_line) > 3 and not any(char in clean_line for char in '/:*?"<>|'):
                    info['nombre_empresa'] = clean_line
                    continue
            for pattern in tax_patterns:
                tax_match = re.search(pattern, clean_line, re.I)
                if tax_match:
                    info['factura'] = tax_match.group(1) if len(tax_match.groups()) > 0 else tax_match.group(0)
                    break
            for pattern in phone_patterns:
                phone_match = re.search(pattern, clean_line, re.I)
                if phone_match:
                    info['telefono'] = phone_match.group(1) if len(phone_match.groups()) > 0 else phone_match.group(0)
                    break

            addr_indicators = ['calle', 'av', 'avenida', 'blvd', 'boulevard', 'plaza', 'col', 'colonia', '#', 'no.', 'número']
            if not info['direccion'] and any(indicator in clean_line.lower() for indicator in addr_indicators):
                info['direccion'] = clean_line

        return info
    
    def _extract_transaction_info(self, text_data: Dict) -> Dict[str, str]:
        info = {
            'transaction_number': None,
            'authorization_number': None,
            'total': None,
            'ruc': None
        }
        for line in text_data['text']:
            clean_line = line.strip()
            if not clean_line:
                continue
            if not info['transaction_number'] and 'transaction' in clean_line.lower():
                info['transaction_number'] = clean_line.split()[-1]
            if not info['authorization_number'] and 'authorization' in clean_line.lower():
                info['authorization_number'] = clean_line.split()[-1]
            if not info['total'] and 'total' in clean_line.lower():
                info['total'] = clean_line.split()[-1]
            if not info['ruc'] and 'ruc' in clean_line.lower():
                info['ruc'] = clean_line.split()[-1]
        return info

    def _extract_payment_info(self, text_data: Dict) -> Dict[str, str]:
        info = {
            
        }
        for line in text_data['text']:
            clean_line = line.strip()
            if not clean_line:
                continue
            if 'visa' in clean_line.lower():
                info['payment_method'] = 'Visa'
            elif 'mastercard' in clean_line.lower():
                info['payment_method'] = 'MasterCard'
            elif 'cash' in clean_line.lower() or 'efectivo' in clean_line.lower():
                info['payment_method'] = 'Cash'
        return info

class ImagePreprocessor:
    """Clase especializada en preprocesamiento de imágenes para OCR con capas específicas"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def preprocess(self, image_path: str, debug: bool = False) -> np.ndarray:
        """Pipeline simplificado de preprocesamiento
        Args:
            image_path: Ruta de la imagen
            debug: Si es True, guarda imágenes intermedias
        Returns:
            Imagen procesada
        """
        # Cargar la imagen
        original = self._load_image(image_path)
        if original is None:
            raise ValueError("No se pudo cargar la imagen")

        if debug:
            cv2.imwrite("debug_original.png", original)

        # Detectar y recortar los bordes del documento en color
        cropped = self._crop_document(original)
        if debug:
            cv2.imwrite("debug_cropped.png", cropped)

        # Convertir la imagen a escala de grises después de recortar
        gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        processed = gray_image.copy()
        steps = [
            ('resize', self._resize_image),
            ('initial_denoise', self._remove_initial_noise),
            ('shadow_removal', self._remove_shadows)
        ]

        for step_name, step_func in steps:
            try:
                processed = step_func(processed)
                if debug:
                    cv2.imwrite(f"debug_{step_name}.png", processed)
            except Exception as e:
                self.logger.warning(f"Error en paso {step_name}: {str(e)}")

        return processed

    def _load_image(self, image_path: str) -> np.ndarray:
        """Carga y realiza verificaciones básicas de la imagen"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen")

        return image

    def _crop_document(self, image: np.ndarray) -> np.ndarray:
        """Detecta y recorta los bordes del documento en color"""
        # Aplicar desenfoque para reducir ruido
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Convertir a escala de grises para detección de bordes
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Detectar bordes
        edged = cv2.Canny(gray, 75, 200)

        # Detectar líneas con la Transformación de Hough
        lines = cv2.HoughLinesP(edged, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            raise ValueError("No se detectaron líneas en la imagen")

        # Encontrar las líneas del contorno del documento
        lines = lines.reshape(-1, 4)
        vertical_lines = []
        horizontal_lines = []
        for x1, y1, x2, y2 in lines:
            if abs(x1 - x2) < 10:  # Línea vertical
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(y1 - y2) < 10:  # Línea horizontal
                horizontal_lines.append((x1, y1, x2, y2))

        if not vertical_lines or not horizontal_lines:
            raise ValueError("No se detectaron líneas válidas en la imagen")

        # Ordenar y tomar las líneas externas para el contorno del documento
        min_x = min(v[0] for v in vertical_lines)
        max_x = max(v[2] for v in vertical_lines)
        min_y = min(h[1] for h in horizontal_lines)
        max_y = max(h[3] for h in horizontal_lines)

        # Obtener los puntos de la esquina del documento
        pts = np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]], dtype="float32")

        # Aplicar la transformación de perspectiva
        cropped = self._four_point_transform(image, pts)

        return cropped

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Aplica la transformación de perspectiva"""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # Calcular las dimensiones de la nueva imagen
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Definir el destino de la imagen transformada
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Obtener la transformación de perspectiva
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Ordena los puntos en orden top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Redimensiona la imagen manteniendo proporción y calidad"""
        height, width = image.shape[:2]
        target_height = 1800
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)

        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        return resized

    def _remove_initial_noise(self, image: np.ndarray) -> np.ndarray:
        """Primera pasada de eliminación de ruido"""
        denoised = cv2.GaussianBlur(image, (3, 3), 0)
        kernel = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        return denoised

    def _remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Elimina sombras usando técnica de morfología"""
        dilated = cv2.dilate(image, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(image, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        return norm

    
class LayeredReceiptOCR:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.layers = [HeaderLayer()]
        self.logger = logging.getLogger(__name__)

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Procesa la imagen del recibo usando todas las capas"""
        try:
            # Preprocesar imagen
            processed_image = self.preprocessor.preprocess(image_path, debug=True)
            return self._process_sequential(processed_image)
        except Exception as e:
            self.logger.error(f"Error procesando imagen: {str(e)}")
            raise

    def _process_sequential(self, image: np.ndarray) -> Dict[str, Any]:
        """Procesa las capas secuencialmente"""
        results = {}

        for layer in self.layers:
            try:
                result = layer.process(image)
                results[layer.name] = asdict(result)
            except Exception as e:
                self.logger.error(f"Error en capa {layer.name}: {str(e)}")
                results[layer.name] = None

        return self._consolidate_results(results)

    def _consolidate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        consolidated = {
            "metadata": {
                "total_confidence": 0.0,
                "processed_layers": 0,
                "successful_layers": 0,
            },
            "data": {},
        }
        total_confidence = 0
        processed_layers = 0
        for layer_name, layer_result in results.items():
            if layer_result is not None:
                processed_layers += 1
                total_confidence += layer_result["confidence"]
                for key, value in layer_result["data"].items():
                    consolidated["data"].setdefault(key, []).append(value)
        if processed_layers > 0:
            consolidated["metadata"]["total_confidence"] = total_confidence / processed_layers
        consolidated["metadata"]["processed_layers"] = len(results)
        consolidated["metadata"]["successful_layers"] = processed_layers
        return consolidated


    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Guarda los resultados en un archivo JSON"""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Resultados guardados en: {output_path}")
        except Exception as e:
            self.logger.error(f"Error guardando resultados: {str(e)}")
            raise


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Crear instancia del procesador
    processor = LayeredReceiptOCR()

    # Procesar imagen
    try:
        results = processor.process_image("deposito.jpeg")
        processor.save_results(results, "output.json")
    except Exception as e:
        logging.error(f"Error en el procesamiento: {str(e)}")