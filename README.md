# Algoritmo Genético 
`(EP1 ACH2053 - Introdução à Estatística 2022)`

## Objetivo
Criar um algoritmo genético modificado para calcular os valores mínimos de algumas funções pré-definidas, gerando gráficos com as médias e o fitness das populações.

## Passo a passo (como genericamente proposto em aula)
1. Definir parâmetros (tamanho da população, probabilidade de crossover, probabilidade de mutação, número de gerações, número de bits, quantidade de variáveis, função, intervalo da função).
2. Gerar população inicial aleatoriamente.
3. Calcular fitness da população.
4. Ordenar população com base no fitness.
5. Calcular média da população e a pessoa com maior fitness.
6. Enquanto não tiver atingido o número de gerações passados por parâmetro:
    - Gera uma população por crossover simples (usando roleta)
    - Gera uma população por crossover uniforme (usando roleta)
    - Gera uma população por mutação clássica (usando roleta)
    - Gera uma população totalmente aleatória
    - Seleciona os indivíduos com mais fitness das populações geradas para integrar a nova população, calcula seu fitness, ordena seus indíviduos pelo fitness e calcula média e melhor indivíduo

## TODO
- Usar o numpy para ter números aleatórios melhores
- Descobrir como fazer funcionar a função que transforma de número inteiro para real (passada em aula)
- Acrescentar a criação de gráficos em cada geração
- Utilizar o código nas funções objetivo
- Refatorar