# IA-NERD
Local PDF reader in Python: extracts text (with OCR fallback), finds relevant passages and summarizes.

O que é o Tesseract

É um programa de OCR (reconhece texto em imagens). O IA NERD usa nas páginas de PDF escaneado, que não têm texto selecionável.
Ele não se instala com pip. O pytesseract (do pip) é só a ponte entre o Python e o programa. Sem o Tesseract instalado no computador, o OCR falha.

Como instalar no Windows

Baixar o instalador do Tesseract para Windows (o mais usado é o da UB Mannheim, que está linkado na própria documentação do Tesseract).
Durante a instalação, marcar o idioma Portuguese em “Additional language data”, se o seu código usa por. O inglês já vem.
O local padrão é C:\Program Files\Tesseract-OCR\tesseract.exe.

Requisitos: Python, as bibliotecas do pip install (pymupdf, pytesseract, pillow, scikit-learn) e o Tesseract.
Como rodar: python minhaIA.py.
Limitação: PDFs escaneados ficam mais lentos por causa do OCR.
