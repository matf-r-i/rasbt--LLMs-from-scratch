import json

with open('ch02.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source']).strip()
        
        if src.startswith('- From these tokens, we can now build a vocabulary'):
            nb['cells'][i]['source'] = ['- Од ових токена сада можемо изградити речник (vocabulary) који се састоји од свих јединствених токена\n']
        elif src.startswith('- Below are the first 50 entries'):
            nb['cells'][i]['source'] = ['- Испод су првих 50 уноса у овом речнику:\n']
        elif src.startswith('- Below, we illustrate the tokenization'):
            nb['cells'][i]['source'] = ['- Испод илуструјемо токенизацију кратког узорка текста користећи мали речник:\n']
        elif src.startswith('- Putting it now all together'):
            nb['cells'][i]['source'] = ['- Сада све то стављамо заједно у класу токенизатора\n']
        elif src.startswith('- The `encode` function turns text into token IDs'):
            nb['cells'][i]['source'] = ['- Функција `encode` претвара текст у ID-ове токена\n', '- Функција `decode` претвара ID-ове токена назад у текст\n']
        elif src.startswith('- We can use the tokenizer to encode'):
            nb['cells'][i]['source'] = ['- Можемо користити токенизатор да кодирамо (тј. токенизирамо) текстове у целе бројеве\n', '- Ови цели бројеви се затим могу уградити (касније) као улаз за LLM\n']
        elif src.startswith('- We can decode the integers'):
            nb['cells'][i]['source'] = ['- Можемо декодирати целе бројеве назад у текст\n']
        elif src.startswith('## 2.4 Adding special context tokens'):
            nb['cells'][i]['source'] = ['## 2.4 Додавање специјалних контекстних токена\n']
        elif src.startswith("- It's useful to add some") and 'special' in src:
            nb['cells'][i]['source'] = ['- Корисно је додати неке специјалне токене за непознате речи и за означавање краја текста\n']
        elif src.startswith('- Some tokenizers use special tokens to help'):
            nb['cells'][i]['source'] = ['- Неки токенизатори користе специјалне токене за додатни контекст\n', '- Неки од њих:\n', '  - `[BOS]` (почетак секвенце)\n', '  - `[EOS]` (крај секвенце)\n', '  - `[PAD]` (допуна)\n', '- `[UNK]` за речи ван речника\n', '\n', '- GPT-2 користи само `<|endoftext|>` токен\n']
        elif src.startswith('- We use the <'):
            pass  # skip this one, already handled by more specific patterns
        elif src.startswith("- Let's see what happens") and 'tokenize' in src:
            nb['cells'][i]['source'] = ['- Да видимо шта се дешава ако токенизирамо следећи текст:\n']
        elif src.startswith('- The above produces an error'):
            nb['cells'][i]['source'] = ['- Горњи код производи грешку јер реч „Hello“ није у речнику\n']
        elif src.startswith('- To deal with such cases'):
            nb['cells'][i]['source'] = ['- За такве случајеве, додајемо `<|unk|>` токен за непознате речи\n']
        elif src.startswith('- Since we are already extending'):
            nb['cells'][i]['source'] = ['- Пошто већ проширујемо речник, додајмо `<|endoftext|>` који означава крај текста\n']
        elif src.startswith('- We also need to adjust the tokenizer'):
            nb['cells'][i]['source'] = ['- Такође прилагодимо токенизатор да користи `<unk>` токен\n']
        elif src.startswith('## 2.5 BytePair encoding'):
            nb['cells'][i]['source'] = ['## 2.5 BytePair кодирање\n']
        elif src.startswith('- GPT-2 used BytePair encoding'):
            nb['cells'][i]['source'] = ['- GPT-2 користи BytePair кодирање (BPE) као токенизатор\n', '- Разлаже непознате речи на подречне јединице\n', '- Користимо `tiktoken` библиотеку\n']
        elif src.startswith('- BPE tokenizers break down'):
            nb['cells'][i]['source'] = ['- BPE токенизатори разлажу непознате речи на подречи и знакове:\n']
        elif src.startswith('## 2.6 Data sampling'):
            nb['cells'][i]['source'] = ['## 2.6 Узорковање података помоћу клизног прозора\n']
        elif src.startswith('- We train LLMs to generate one word'):
            nb['cells'][i]['source'] = ['- Тренирамо LLM да генерише реч по реч, где следећа реч представља циљ:\n']
        elif src.startswith('- For each text chunk'):
            nb['cells'][i]['source'] = ['- За сваки део желимо улазе и циљеве\n', '- Циљеви су улази померени за 1 позицију удесно\n']
        elif src.startswith('- One by one, the prediction'):
            nb['cells'][i]['source'] = ['- Предвиђање би изгледало:\n']
        elif src.startswith('- We will take care of the next-word prediction'):
            nb['cells'][i]['source'] = ['- Предикцијом следеће речи бавићемо се касније\n', '- Сада имплементирамо data loader\n']
        elif src.startswith('- Install and import PyTorch'):
            nb['cells'][i]['source'] = ['- Инсталирајте PyTorch (види Додатак А)\n']
        elif src.startswith('- We use a sliding window approach'):
            nb['cells'][i]['source'] = ['- Користимо клизни прозор, померајући за +1:\n']
        elif src.startswith('- Create dataset and dataloader'):
            nb['cells'][i]['source'] = ['- Креирамо dataset и dataloader\n']
        elif src.startswith("- Let's test the dataloader"):
            nb['cells'][i]['source'] = ['- Тестирамо dataloader са batch size 1 и context size 4:\n']
        elif src.startswith('- An example using stride equal to'):
            nb['cells'][i]['source'] = ['- Пример са stride = context length (4):\n']
        elif src.startswith('- We can also create batched outputs'):
            nb['cells'][i]['source'] = ['- Такође креирамо пакетне излазе\n', '- Повећавамо stride да избегнемо overfitting\n']
        elif src.startswith('## 2.7 Creating token embeddings'):
            nb['cells'][i]['source'] = ['## 2.7 Креирање векторских репрезентација токена (token embeddings)\n']
        elif src.startswith('- The data is already almost ready'):
            nb['cells'][i]['source'] = ['- Подаци су скоро спремни за LLM\n', '- Уградимо токене у континуалну векторску репрезентацију\n']
        elif src.startswith('- Suppose we have the following four input examples'):
            nb['cells'][i]['source'] = ['- Претпоставимо 4 улаза са ID 2, 3, 5, 1:\n']
        elif src.startswith('- For the sake of simplicity'):
            nb['cells'][i]['source'] = ['- Речник од 6 речи, embeddings величине 3:\n']
        elif src.startswith('- This would result in a 6x3'):
            nb['cells'][i]['source'] = ['- Матрица тежина 6x3:\n']
        elif src.startswith('- For those who are familiar with one-hot encoding'):
            nb['cells'][i]['source'] = ['- Embedding layer је ефикаснија имплементација one-hot кодирања\n']
        elif src.startswith('- Because the embedding layer is just'):
            nb['cells'][i]['source'] = ['- Може се оптимизовати путем backpropagation-а\n']
        elif src.startswith('- To convert a token with id 3'):
            nb['cells'][i]['source'] = ['- Конвертујемо токен ID 3 у 3D вектор:\n']
        elif src.startswith('- Note that the above is the 4th row'):
            nb['cells'][i]['source'] = ['- Ово је 4. ред у `embedding_layer` матрици\n']
        elif src.startswith('- To embed all four'):
            nb['cells'][i]['source'] = ['- Да уградимо све `input_ids`:\n']
        elif src.startswith('- An embedding layer is essentially a look-up operation'):
            nb['cells'][i]['source'] = ['- Embedding layer је операција претраживања (look-up):\n']
        elif src.startswith('- **You may be interested'):
            nb['cells'][i]['source'] = ['- **Поређење embedding-а са линеарним слојевима: [../03_bonus_embedding-vs-matmul](../03_bonus_embedding-vs-matmul)**\n']
        elif src.startswith('## 2.8 Encoding word positions'):
            nb['cells'][i]['source'] = ['## 2.8 Кодирање позиција речи\n']
        elif src.startswith('- Embedding layer convert IDs into identical'):
            nb['cells'][i]['source'] = ['- Embedding layer даје исте векторе без обзира на позицију:\n']
        elif src.startswith('- Positional embeddings are combined'):
            nb['cells'][i]['source'] = ['- Позициони embeddings се комбинују са token embeddings:\n']
        elif src.startswith('- The BytePair encoder has a vocabulary size of 50,257'):
            nb['cells'][i]['source'] = ['- BytePair кодер има речник од 50.257:\n', '- Желимо 256-димензионалне репрезентације:\n']
        elif src.startswith('- If we sample data from the dataloader'):
            nb['cells'][i]['source'] = ['- Узоркујемо податке у 256-димензионални вектор\n', '- Са batch size 8 и 4 токена: тензор 8 x 4 x 256:\n']
        elif src.startswith('- GPT-2 uses absolute'):
            nb['cells'][i]['source'] = ['- GPT-2 користи апсолутне позиционе embeddings:\n']
        elif src.startswith('- To create the input embeddings used in an LLM'):
            nb['cells'][i]['source'] = ['- Сабирамо token и позиционе embeddings за улаз у LLM:\n']
        elif src.startswith('- In the initial phase'):
            nb['cells'][i]['source'] = ['- У почетној фази, текст се сегментира у токене\n', '- Затим се претварају у ID-ове:\n']
        elif src.startswith('# Summary and takeaways'):
            nb['cells'][i]['source'] = ['# Резиме и закључци\n']
        elif src.startswith('See the [./dataloader.ipynb]'):
            nb['cells'][i]['source'] = ['Погледајте [./dataloader.ipynb](./dataloader.ipynb) - сажету верзију.\n', '\n', 'Погледајте [./exercise-solutions.ipynb](./exercise-solutions.ipynb) за решења.\n']
        elif src.startswith("Let's try to tokenize text"):
            nb['cells'][i]['source'] = ['Пробајмо да токенизирамо текст са измењеним токенизатором:\n']

# Translate code comments
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        new_src = []
        for line in cell['source']:
            if line.strip().startswith('# Strip whitespace'):
                new_src.append('    # Уклони белине из сваке ставке, затим филтрирај празне стрингове.\n')
            elif line.strip().startswith('# Replace spaces'):
                new_src.append('        # Замени размаке испред интерпункцијских знакова\n')
            elif 'print("Inputs:' in line and 'inputs' in line:
                new_src.append('print("Улази (Inputs):\\n", inputs)\n')
            elif 'print("\\nTargets:' in line:
                new_src.append('print("\\nЦиљеви (Targets):\\n", targets)\n')
            elif 'print("Total number of character:"' in line:
                new_src.append('print("Укупан број знакова:", len(raw_text))\n')
            else:
                new_src.append(line)
        nb['cells'][i]['source'] = new_src

with open('ch02.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Превод је завршен!')
