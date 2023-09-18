# Adicionar título do artigo

- Trabalho baseado no trabalho de ...
- Obs: Para executar o experimento para o whatsapp, basta trocar diretórios e arquivos com final '_telegram' para '_whatsapp'.

## Tecnologia
- Python 3.9

## Configuração de Ambiente local

- Clonar o projeto

```bash
  $ git clone https://github.com/jmmfilho/faroldigital-etl.git
```

- Criar e ativar a virtualenv

```bash
    `./$ virtualenv -p=python3 venv-experimento-sbseg`
    `./$ source venv-experimento-sbseg/bin/activate`
```

- Instalar as dependências

```bash
    `(venv-experimento-sbseg) ./$ pip install -r requirements.txt`
```

-  Instalar NLTK stopwords
```bash 
   `./$ python -m nltk.downloader stopwords`
```

- Executar a aplicação

```bash
  ./scripts$ python3 experimento.py
```