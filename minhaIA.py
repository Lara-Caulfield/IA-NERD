"""IA NERD: busca e resumo em PDFs, tudo local.

Uso:
    python ia_nerd.py                 # abre uma janela para você escolher o PDF
    python ia_nerd.py meu_livro.pdf   # já abre esse PDF
    (dentro do programa, 'abrir' troca de PDF a qualquer momento)

Organização das matérias (uma pasta por matéria, com os PDFs dentro):
    materias/
        Seguranca de Sistemas/   sistemas1.pdf, sistemas2.pdf ...
        Outra Materia/           ...
Se não existir a pasta 'materias', usa as subpastas ao lado do script
(pastas cujo nome começa com "tesseract" são ignoradas).
Para apontar outro lugar: variável de ambiente IA_NERD_MATERIAS.

Dependências:
    pip install pymupdf pytesseract pillow scikit-learn numpy
    (e o Tesseract instalado no Windows)
"""

import argparse
import os
import re
import textwrap

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# O caminho do Tesseract pode vir da variável de ambiente TESSERACT_CMD.
TESSERACT_PADRAO = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", TESSERACT_PADRAO)

PASTA = os.path.dirname(os.path.abspath(__file__))


# ---------- Leitura do PDF ----------

def limpar_texto(texto):
    """Limpa o texto sem destruir os parágrafos."""
    texto = texto.replace("-\n", "")                  # junta palavras quebradas por hífen
    texto = re.sub(r"[ \t]+", " ", texto)             # espaços repetidos
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)        # parágrafos = linha em branco
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)    # quebra simples vira espaço
    return texto.strip()


def texto_da_pagina(pagina):
    """Texto nativo da página, um bloco por parágrafo."""
    blocos = pagina.get_text("blocks", sort=True)
    textos = [b[4].strip() for b in blocos if b[6] == 0 and b[4].strip()]
    return "\n\n".join(textos)


def ocr_da_pagina(pagina):
    """OCR em memória (sem arquivo temporário)."""
    pix = pagina.get_pixmap(matrix=fitz.Matrix(2, 2))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    return pytesseract.image_to_string(img, lang="eng+por", config="--psm 6")


def extrair_paginas(caminho_pdf):
    """Devolve uma lista de (numero_da_pagina, texto)."""
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"PDF não encontrado: {caminho_pdf}")

    paginas = []
    with fitz.open(caminho_pdf) as doc:
        for numero, pagina in enumerate(doc, start=1):
            texto = texto_da_pagina(pagina)

            if texto:
                print(f"Página {numero}: texto encontrado")
            else:
                print(f"Página {numero}: usando OCR...")
                texto = ocr_da_pagina(pagina)

            texto = limpar_texto(texto)
            if texto:
                paginas.append((numero, texto))

    return paginas


# ---------- Trechos e busca ----------

def quebrar_texto(texto, tamanho=250, minimo=15):
    """Agrupa parágrafos em trechos de ~tamanho palavras, repetindo o último
    parágrafo de cada trecho no começo do próximo (sobreposição)."""
    paragrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    trechos, atual, contagem = [], [], 0

    for p in paragrafos:
        n = len(p.split())

        if atual and contagem + n > tamanho:
            trechos.append("\n\n".join(atual))
            atual = [atual[-1]]
            contagem = len(atual[0].split())

        atual.append(p)
        contagem += n

    if atual:
        trechos.append("\n\n".join(atual))

    return [t for t in trechos if len(t.split()) >= minimo]


def criar_trechos(paginas):
    trechos = []
    for numero, texto in paginas:
        for parte in quebrar_texto(texto):
            trechos.append({"pagina": numero, "texto": parte})
    return trechos


def buscar(pergunta, vectorizer, matriz, trechos, quantidade=3):
    vetor = vectorizer.transform([pergunta])
    similaridades = cosine_similarity(vetor, matriz).flatten()
    melhores = similaridades.argsort()[-quantidade:][::-1]
    return [(trechos[i], similaridades[i]) for i in melhores if similaridades[i] > 0]


# ---------- Resumo ----------

def dividir_frases(texto):
    frases = re.split(r"(?<=[.!?])\s+", texto.replace("\n", " "))
    return [f.strip() for f in frases if len(f.split()) >= 6]


def resumir(texto, n_frases=8):
    """Resumo por extração: escolhe as frases mais parecidas com o 'centro'
    do texto e as mostra na ordem original."""
    frases = dividir_frases(texto)
    if len(frases) <= n_frases:
        return frases

    matriz = TfidfVectorizer(lowercase=True).fit_transform(frases)
    centro = np.asarray(matriz.mean(axis=0))
    pontuacoes = cosine_similarity(matriz, centro).flatten()
    indices = sorted(pontuacoes.argsort()[-n_frases:])
    return [frases[i] for i in indices]


# ---------- Saída bonita ----------

def mostrar_texto(texto, largura=100):
    for paragrafo in texto.split("\n\n"):
        print(textwrap.fill(paragrafo, width=largura))
        print()


def mostrar_resumo(texto):
    frases = resumir(texto)
    if not frases:
        print("Texto curto demais para resumir.")
        return

    print("\nResumo:\n")
    for frase in frases:
        print(textwrap.fill(f"- {frase}", width=100, subsequent_indent="  "))
    print()


# ---------- Escolher e carregar PDF ----------

def pasta_das_materias():
    """Pasta onde ficam as matérias (uma subpasta por matéria).
    Ordem: variável IA_NERD_MATERIAS, subpasta 'materias' ao lado do script,
    ou a própria pasta do script."""
    env = os.environ.get("IA_NERD_MATERIAS")
    if env and os.path.isdir(env):
        return env
    padrao = os.path.join(PASTA, "materias")
    return padrao if os.path.isdir(padrao) else PASTA


# Pastas que nunca viram "matéria" (ex.: código do Tesseract baixado do GitHub).
IGNORAR = ("tesseract", "node_modules", "venv", ".git")
SOLTOS = "(PDFs soltos nesta pasta)"


def listar_pdfs(pasta, recursivo=False):
    """Devolve os PDFs como caminhos relativos a 'pasta', em ordem alfabética."""
    achados = []
    if recursivo:
        for atual, subpastas, arquivos in os.walk(pasta):
            subpastas[:] = [d for d in subpastas if not d.lower().startswith(IGNORAR)]
            for f in arquivos:
                if f.lower().endswith(".pdf"):
                    achados.append(os.path.relpath(os.path.join(atual, f), pasta))
    else:
        achados = [f for f in os.listdir(pasta) if f.lower().endswith(".pdf")]
    return sorted(achados, key=str.lower)


def listar_materias(raiz):
    """Subpastas com PDFs (inclusive dentro de subpastas delas), mais a
    opção de PDFs soltos na própria pasta."""
    materias = []
    for nome in sorted(os.listdir(raiz), key=str.lower):
        caminho = os.path.join(raiz, nome)
        if (os.path.isdir(caminho)
                and not nome.startswith((".", "__"))
                and not nome.lower().startswith(IGNORAR)
                and listar_pdfs(caminho, recursivo=True)):
            materias.append(nome)
    if listar_pdfs(raiz):
        materias.append(SOLTOS)
    return materias


def perguntar_numero(titulo, opcoes, extra=None):
    """Mostra uma lista numerada e devolve o índice escolhido (ou None).
    'extra' é uma opção especial exibida como 0."""
    print(f"\n{titulo}")
    if extra:
        print(f"  0. {extra}")
    for i, nome in enumerate(opcoes, start=1):
        print(f"  {i}. {nome}")
    resposta = input("Número (Enter para cancelar): ").strip()
    if resposta == "0" and extra:
        return 0
    if resposta.isdigit() and 1 <= int(resposta) <= len(opcoes):
        return int(resposta)
    return None


def escolher_arquivo_manual():
    """Janela do Windows para escolher qualquer PDF (ou digitar o caminho)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        raiz = tk.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        caminho = filedialog.askopenfilename(title="Escolha o PDF", filetypes=[("PDF", "*.pdf")])
        raiz.destroy()
        return caminho or None
    except Exception:
        caminho = input("Digite o caminho do PDF (Enter para cancelar): ").strip().strip('"')
        return caminho or None


def escolher_pdf():
    """Menu: matéria -> PDF. Devolve o caminho do PDF ou None."""
    raiz = pasta_das_materias()
    materias = listar_materias(raiz)

    # Sem subpastas: lista os PDFs soltos na própria pasta.
    if not materias:
        pdfs = listar_pdfs(raiz)
        if not pdfs:
            print(f"Não achei PDFs em: {raiz}")
            return escolher_arquivo_manual()
        n = perguntar_numero("PDFs disponíveis:", pdfs, extra="Outro arquivo...")
        if n == 0:
            return escolher_arquivo_manual()
        return os.path.join(raiz, pdfs[n - 1]) if n else None

    while True:
        n = perguntar_numero("Matérias:", materias, extra="Outro arquivo...")
        if n is None:
            return None
        if n == 0:
            return escolher_arquivo_manual()

        soltos = materias[n - 1] == SOLTOS
        pasta = raiz if soltos else os.path.join(raiz, materias[n - 1])
        pdfs = listar_pdfs(pasta, recursivo=not soltos)
        p = perguntar_numero(f"PDFs de {materias[n - 1]}:", pdfs, extra="Voltar")
        if p:
            return os.path.join(pasta, pdfs[p - 1])
        # 0 (voltar) ou Enter: volta para a lista de matérias


def carregar(caminho):
    """Lê o PDF e monta a base de busca. Devolve um dicionário ou None."""
    try:
        paginas = extrair_paginas(caminho)
    except Exception as erro:
        print(f"Não consegui abrir esse PDF: {erro}")
        return None

    trechos = criar_trechos(paginas)
    if not trechos:
        print("Não consegui extrair texto suficiente desse PDF.")
        return None

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2))
    matriz = vectorizer.fit_transform([t["texto"] for t in trechos])

    print(f"\nBase carregada: {os.path.basename(caminho)} ({len(trechos)} trechos)")
    return {
        "caminho": caminho,
        "paginas": paginas,
        "trechos": trechos,
        "vectorizer": vectorizer,
        "matriz": matriz,
    }


# ---------- Programa principal ----------

def main():
    parser = argparse.ArgumentParser(description="IA NERD: busca e resumo em PDFs")
    parser.add_argument("pdf", nargs="?", default=None,
                        help="caminho do PDF (se não informar, você escolhe)")
    args = parser.parse_args()

    # Nenhum PDF fixo: começa vazio e só preenche quando você escolher.
    base = None
    if args.pdf:
        caminho = args.pdf if os.path.exists(args.pdf) else os.path.join(PASTA, args.pdf)
        base = carregar(caminho)

    while base is None:
        caminho = escolher_pdf()
        if not caminho:
            print("Nenhum PDF escolhido. Saindo.")
            return
        base = carregar(caminho)

    while True:
        entrada = input(
            "\nPergunta, 'resumo', 'resumo N' (página), 'abrir' (trocar PDF) ou 'sair': "
        ).strip()
        comando = entrada.lower()

        if comando == "sair":
            break
        if not entrada:
            continue

        if comando == "abrir":
            caminho = escolher_pdf()
            if caminho:
                nova = carregar(caminho)
                if nova:
                    base = nova  # só troca se o novo PDF carregou direito
            continue

        pedido_resumo = re.fullmatch(r"resumo(?:\s+(\d+))?", comando)
        if pedido_resumo:
            if pedido_resumo.group(1):
                alvo = int(pedido_resumo.group(1))
                texto = " ".join(t for n, t in base["paginas"] if n == alvo)
                if not texto:
                    print("Essa página não existe ou não tem texto.")
                    continue
            else:
                texto = " ".join(t for _, t in base["paginas"])
            mostrar_resumo(texto)
            continue

        resultados = buscar(entrada, base["vectorizer"], base["matriz"], base["trechos"])
        if not resultados:
            print("Não encontrei nada muito relacionado com essa pergunta.")
            continue

        print("\nResposta baseada no PDF:\n")
        for i, (trecho, score) in enumerate(resultados, start=1):
            print(f"--- Trecho {i} | página {trecho['pagina']} | relevância: {score:.2f} ---\n")
            mostrar_texto(trecho["texto"])


if __name__ == "__main__":
    main()