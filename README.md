# BERTweetBR: Construção de um Modelo de Linguagem para Análise de Tweets em Português

O objetivo principal deste projeto[^1] é trazer contribuições científicas relevantes para o desenvolvimento e aplicação prática de um modelo BERT-like especializado em textos curtos e informais em português que possa ser usado para análise de tweets relacionados/direcionados a eventos, como a FEBRACE.

Objetivos específicos deste projeto incluem:

- Estudar os aspectos teóricos e práticos interdisciplinares relacionados à modelagem BERT-like para Processamento de Linguagem Natural, considerando inclusive aspectos jurídicos, éticos, sociais e tecnológicos. 
- Coletar e armazenar dados de treinamento em forma de textos curtos em português, especificamente Tweets.
- Criar e treinar um modelo BERT-like para análise de tweets, com propósito geral, que poderá ser utilizado para diversas tarefas no contexto de redes sociais em português. 
- Utilizar este modelo para a análise de tweets relativos a FEBRACE. 
- Estudar aspectos de robustez deste modelo a entradas inesperadas, bem como aspectos que podem ser transferidos para outras línguas. 

# Como utilizar

É necessário usar o modelo BERTimbau como base de treino. Instale as dependências necessárias e então rode o arquivo bertweetbr_train (ajuste as variáveis no início do arquivo de acordo com seu objetivo). Ele vai gerar uma pasta com o modelo treinado.

Para avaliar o modelo, rode o arquivo bertweetbr_eval (novamente, ajuste as variáveis no início do arquivo de acordo com seu objetivo. Será necessário atualizar o caminho do modelo para o seu modelo treinado). Ele vai gerar uma pasta com um arquivo .txt com as métricas salvas, e também um arquivo .out.tf.events para ser aberto com o TensorBoard.

Para coletar os tweets, basta preencher o arquivo TweetCollectBySearch com os valores apropriados da sua chave de API v2 do Twitter, uma keyword de busca se desejável, um limite de tweets para coleta e o destino desses tweets. Os tweets vão ser salvo em .csv mas caso queira usá-los como .txt basta rodar o arquivo csvToText com o número apropriado no limite de leitura do laço de repetição.


[^1]: Plano de pesquisa faz parte do projeto aprovado no Edital: MCTIC/CNPq N° 05/2019, Processo nº: 441081/2019-3 “Avaliação de impacto da adoção de abordagens envolvendo iniciação à pesquisa e participação em feiras investigativas na Educação Básica, por meio de aplicação de Visual Analytics & Learning Analytics”.
