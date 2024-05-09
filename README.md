# ChatBot para editais
Este é um aplicativo de chat baseado em Streamlit que permite aos usuários fazer perguntas sobre seus editais carregados em formato PDF. O aplicativo usa processamento de linguagem natural (NLP) para fornecer respostas relevantes às perguntas dos usuários.

## Funcionalidades
- Carregar vários editais em formato PDF.
- Fazer perguntas sobre os editais carregados.
- Receber respostas relevantes com base no conteúdo dos editais.
- Conversar com um modelo de linguagem treinado para fornecer respostas contextualizadas.


## Instalação

Para instalar o Ollama e baixar o modelo Llama3, siga estas etapas:

### 1. Baixar e instalar o Ollama

- Visite o site oficial do [Ollama](https://ollama.com/) para baixar o instalador adequado para o seu sistema operacional.

### 2. Baixar o modelo Llama3

- Após baixar e instalar o Ollama, abra um terminal ou prompt de comando.

- Execute o seguinte comando para baixar o modelo Llama3:

```
ollama run llama3
```

Isso fará o download do modelo Llama3 para uso com o Ollama.

### 3. Baixar o modelo Nomic-embed-text

- Com o Ollama já instalado, execute o seguinte comando para baixar o modelo Nomic-embed-text:

```
ollama pull nomic-embed-text
```

Isso fará o download do modelo Nomic-embed-text para uso com o Ollama.

 ### 4. Instalar as dependências do projeto:
- No diretório raiz do seu projeto, execute o seguinte comando para instalar as dependências listadas no arquivo `requirements.txt`:

```
pip install -r requirements.txt
```
Isso instalará todas as dependências necessárias para o projeto usando o pip.

## Uso

Para iniciar o aplicativo, execute o seguinte comando:

```
streamlit run main.py
```
Isso abrirá o aplicativo em seu navegador padrão.

- Carregue seus editais em formato PDF clicando no botão "Carregar editais" na barra lateral e selecionando os arquivos desejados.
- Clique em "Processar" para processar os editais carregados.
- Faça perguntas sobre os editais na caixa de entrada de chat e pressione Enter para enviar.
- Receba respostas relevantes na área de chat.
  


