import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator

# Baixa o léxico do VADER (necessário apenas na primeira execução)
nltk.download('vader_lexicon')

# Inicializa o analisador de sentimentos do NLTK
sia = SentimentIntensityAnalyzer()

# Interface do Streamlit
st.title("Análise de Sentimentos")
st.write("Digite um texto em português para descobrir seu sentimento geral.")

# Recebe o texto do usuário em português
texto_pt = st.text_area("Texto em português:")

# Botão para disparar a análise
if st.button("Analisar"):
    if texto_pt.strip():
        # O VADER funciona nativamente em inglês. 
        # Traduzimos o texto de PT para EN de forma transparente antes da análise.
        texto_en = GoogleTranslator(source='pt', target='en').translate(texto_pt)
        
        # Analisa o sentimento do texto traduzido
        scores = sia.polarity_scores(texto_en)
        compound = scores['compound']
        
        # Classifica o resultado com base no score compound
        if compound >= 0.05:
            label = "Positivo"
        elif compound <= -0.05:
            label = "Negativo"
        else:
            label = "Neutro"
            
        # Exibe o resultado na tela
        st.subheader("Resultado")
        st.write(f"**Sentimento:** {label}")
        st.write(f"**Score Compound:** {compound}")
    else:
        st.warning("Por favor, digite um texto.")