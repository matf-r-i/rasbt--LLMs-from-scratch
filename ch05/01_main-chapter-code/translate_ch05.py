import json

with open('ch05.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source']).strip()
        
        # --- Header ---
        if src.startswith('<table'):
            nb['cells'][i]['source'] = [
                '<table style=\"width:100%\">\n',
                '<tr>\n',
                '<td style=\"vertical-align:middle; text-align:left;\">\n',
                '<font size=\"2\">\n',
                'Додатни код за књигу <a href=\"http://mng.bz/orYv\">Build a Large Language Model From Scratch</a> аутора <a href=\"https://sebastianraschka.com\">Sebastian Raschka</a><br>\n',
                '<br>Репозиторијум кода: <a href=\"https://github.com/rasbt/LLMs-from-scratch\">https://github.com/rasbt/LLMs-from-scratch</a>\n',
                '</font>\n',
                '</td>\n',
                '<td style=\"vertical-align:middle; text-align:left;\">\n',
                '<a href=\"http://mng.bz/orYv\"><img src=\"https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp\" width=\"100px\"></a>\n',
                '</td>\n',
                '</tr>\n',
                '</table>'
            ]
        elif src.startswith('# Chapter 5: Pretraining on Unlabeled Data'):
            nb['cells'][i]['source'] = ['# Поглавље 5: Предтренирање на неозначеним подацима\n']
        elif src.startswith('from importlib.metadata'):
            nb['cells'][i]['source'] = ['\n']
        elif src.startswith('- In this chapter, we implement the training loop'):
            nb['cells'][i]['source'] = ['- У овом поглављу имплементирамо петљу тренирања и код за основну евалуацију модела за предтренирање LLM-а:\n', '- На крају поглавља, такође учитавамо јавно доступне предтрениране тежине OpenAI-ја у наш модел:\n']
        elif src.startswith('- The topics covered in this chapter'):
            nb['cells'][i]['source'] = ['- Теме које су покривене у овом поглављу приказане су испод:\n']
            
        # --- 5.1 ---
        elif src.startswith('## 5.1 Evaluating generative text models'):
            nb['cells'][i]['source'] = ['## 5.1 Евалуација генеративних текст модела\n']
        elif src.startswith('- We start this section with a brief recap'):
            nb['cells'][i]['source'] = ['- Почињемо овај одељак кратким резимеом иницијализације GPT модела користећи код из претходног поглавља:\n', '- Затим, дискутујемо основне евалуационе метрике за LLM-ове:\n', '- На крају, примењујемо ове метрике на тренинг и валидациони скуп података:\n']
        elif src.startswith('### 5.1.1 Using GPT to generate text'):
            nb['cells'][i]['source'] = ['### 5.1.1 Коришћење GPT за генерисање текста\n']
        elif src.startswith('- We initialize a GPT model using the code from the previous chapter'):
            nb['cells'][i]['source'] = ['- Иницијализујемо GPT модел користећи код из претходног поглавља:\n']
        elif src.startswith('- We use dropout of 0.1 above'):
            nb['cells'][i]['source'] = ['- Користимо dropout од 0.1 горе, али данас је релативно уобичајено тренирати LLM-ове без dropout-а:\n', '- Модерни LLM-ови такође не користе bias векторе у `nn.Linear` слојевима за query, key и value матрице:\n', '- Смањили смо `context_length` на 256 токена да бисмо смањили рачунске захтеве:\n']
        elif src.startswith('- Next, we use the `generate_text_simple`'):
            nb['cells'][i]['source'] = ['- Затим користимо функцију `generate_text_simple` из претходног поглавља за генерисање текста:\n', '- Такође дефинишемо две помоћне функције, `text_to_token_ids` и `token_ids_to_text`:\n']
        elif src.startswith('- As we can see above, the model does not'):
            nb['cells'][i]['source'] = ['- Као што видимо изнад, модел не производи добар текст јер још увек није трениран:\n', '- Како да нумерички измеримо или ухватимо шта је „добар текст“?\n', '- Следећи пододељак уводи метрике за израчунавање loss-а за генерисане излазе:\n']
            
        # --- 5.1.2 ---
        elif src.startswith('### 5.1.2 Calculating the text generation loss'):
            nb['cells'][i]['source'] = ['### 5.1.2 Израчунавање loss-а за генерисање текста: унакрсна ентропија и перплексија\n']
        elif src.startswith("- Suppose we have an `inputs` tensor"):
            nb['cells'][i]['source'] = ['- Претпоставимо да имамо `inputs` тензор који садржи ID-ове токена за 2 примера за тренирање (редова):\n', '- У односу на `inputs`, `targets` садрже жељене ID-ове токена које желимо да модел генерише:\n', '- Приметите да су `targets` заправо `inputs` померени за 1 позицију:\n']
        elif src.startswith('- Feeding the `inputs` to the model'):
            nb['cells'][i]['source'] = ['- Прослеђивањем `inputs` моделу, добијамо logits вектор за 2 улазна примера:\n', '- Сваки од токена је 50.257-димензионални вектор који одговара величини речника:\n', '- Применом softmax функције, претварамо logits тензор у тензор истих димензија који садржи вероватноће:\n']
        elif src.startswith('- The figure below, using a very small vocabulary'):
            nb['cells'][i]['source'] = ['- Слика испод, користећи веома мали речник за илустрацију, приказује како конвертујемо вероватноће назад у текст:\n']
        elif src.startswith('- As discussed in the previous chapter'):
            nb['cells'][i]['source'] = ['- Као што је дискутовано у претходном поглављу, примењујемо `argmax` функцију за конверзију вероватноћа у предвиђене ID-ове токена:\n']
        elif src.startswith('- Since we have 2 input batches'):
            nb['cells'][i]['source'] = ['- Пошто имамо 2 улазна пакета са по 3 токена, добијамо 2 x 3 предвиђена ID-а токена:\n']
        elif src.startswith('- If we decode these tokens'):
            nb['cells'][i]['source'] = ['- Ако декодирамо ове токене, откривамо да су прилично различити од токена које желимо да модел предвиди:\n']
        elif src.startswith("- That's because the model wasn't trained yet"):
            nb['cells'][i]['source'] = ['- То је зато што модел још увек није трениран:\n', '- Да бисмо тренирали модел, морамо знати колико је удаљен од тачних предвиђања (циљева):\n']
        elif src.startswith('- The token probabilities corresponding'):
            nb['cells'][i]['source'] = ['- Вероватноће токена које одговарају циљним индексима су следеће:\n']
        elif src.startswith('- We want to maximize all these values'):
            nb['cells'][i]['source'] = ['- Желимо да максимизујемо све ове вредности, приближавајући их вероватноћи 1:\n', '- У математичкој оптимизацији, лакше је максимизовати логаритам вероватноће него саму вероватноћу:\n']
        elif src.startswith('# Compute logarithm'):
            pass  # skip code comments
        elif src.startswith('- Next, we compute the average log probability'):
            nb['cells'][i]['source'] = ['- Затим израчунавамо просечну логаритамску вероватноћу:\n']
        elif src.startswith('- The goal is to make this average'):
            nb['cells'][i]['source'] = ['- Циљ је да ову просечну логаритамску вероватноћу учинимо што већом оптимизацијом тежина модела:\n', '- Због логаритма, највећа могућа вредност је 0, а тренутно смо далеко од 0:\n']
        elif src.startswith('- In deep learning, instead of maximizing'):
            nb['cells'][i]['source'] = ['- У дубоком учењу, уместо максимизације просечне лог-вероватноће, стандардно је минимизовати *негативну* просечну лог-вероватноћу:\n', '- Вредност негативне просечне лог-вероватноће назива се унакрсна ентропија (cross-entropy loss):\n']
        elif src.startswith('- PyTorch already implements'):
            nb['cells'][i]['source'] = ['- PyTorch већ имплементира `cross_entropy` функцију која обавља претходне кораке:\n']
        elif src.startswith('- Before we apply the `cross_entropy` function'):
            nb['cells'][i]['source'] = ['- Пре него што применимо `cross_entropy` функцију, проверимо облик logits и targets тензора:\n']
        elif src.startswith('- For the `cross_entropy` function in PyTorch'):
            nb['cells'][i]['source'] = ['- За `cross_entropy` функцију у PyTorch-у, желимо да спљоштимо (flatten) ове тензоре комбиновањем преко batch димензије:\n']
        elif src.startswith('- Note that the targets are the token IDs'):
            nb['cells'][i]['source'] = ['- Имајте на уму да су targets ID-ови токена, који такође представљају индексне позиције у logits тензорима:\n', '- Функција `cross_entropy` ће аутоматски применити softmax и логаритамску вероватноћу интерно:\n']
        elif src.startswith('- A concept related to the cross-entropy loss'):
            nb['cells'][i]['source'] = ['- Концепт повезан са cross-entropy loss-ом је перплексија (perplexity) LLM-а:\n', '- Perplexity је једноставно експоненцијална функција cross-entropy loss-а:\n']
        elif src.startswith('- The perplexity is often considered'):
            nb['cells'][i]['source'] = ['- Perplexity се често сматра интерпретабилнијим јер се може разумети као ефективна величина речника у коју модел није сигуран:\n', '- Слично loss-у, нижа perplexity означава да су предвиђања модела ближа стварној расподели:\n']
            
        # --- 5.1.3 ---
        elif src.startswith('### 5.1.3 Calculating the training and validation set losses'):
            nb['cells'][i]['source'] = ['### 5.1.3 Израчунавање loss-а на тренинг и валидационом скупу\n']
        elif src.startswith('- We use a relatively small dataset'):
            nb['cells'][i]['source'] = ['- Користимо релативно мали скуп података за тренирање LLM-а (само једну кратку причу):\n', '- Разлози су:\n', '  - Можете покренути примере кода за неколико минута на лаптопу без одговарајућег GPU-а\n', '  - Тренирање се завршава релативно брзо (минути уместо недеља)\n', '  - Користимо текст из јавног домена:\n']
        elif src.startswith('- For example, Llama 2 7B'):
            nb['cells'][i]['source'] = ['- На пример, Llama 2 7B је захтевала 184.320 GPU сати на A100 GPU-овима за тренирање на 2 трилиона токена:\n', '- Цена 8xA100 облак сервера на AWS је приближно $30/сат\n', '- Дакле, тренирање овог LLM-а би коштало 184.320 / 8 * $30 = ~$690.000:\n']
        elif src.startswith('- Below, we use the same dataset'):
            nb['cells'][i]['source'] = ['- Испод користимо исти скуп података из поглавља 2:\n']
        elif src.startswith('- A quick check that the text loaded ok'):
            nb['cells'][i]['source'] = ['- Брза провера да је текст успешно учитан:\n']
        elif src.startswith('- With 5,145 tokens, the text is very short'):
            nb['cells'][i]['source'] = ['- Са 5.145 токена, текст је веома кратак за тренирање LLM-а, али у образовне сврхе:\n']
        elif src.startswith('- Next, we divide the dataset'):
            nb['cells'][i]['source'] = ['- Затим делимо скуп података на тренинг и валидациони скуп и користимо data loader-е из поглавља 2:\n', '- Слика испод претпоставља `max_length=6` за визуелизацију:\n']
        elif src.startswith('- We use a relatively small batch size'):
            nb['cells'][i]['source'] = ['- Користимо релативно малу величину пакета (batch size) да смањимо рачунске захтеве:\n', '- Llama 2 7B је тренирана са batch size од 1024, на пример:\n']
        elif src.startswith('- An optional check that the data'):
            nb['cells'][i]['source'] = ['- Опциона провера да су подаци исправно учитани:\n']
        elif src.startswith('- Another optional check'):
            nb['cells'][i]['source'] = ['- Још једна опциона провера да су величине токена у очекиваном опсегу:\n']
        elif src.startswith('- Next, we implement a utility function'):
            nb['cells'][i]['source'] = ['- Затим имплементирамо помоћну функцију за израчунавање cross-entropy loss-а датог пакета:\n', '- Такође имплементирамо другу помоћну функцију за израчунавање loss-а за одређени број пакета:\n']
        elif src.startswith('- If you have a machine with a CUDA-supported GPU'):
            nb['cells'][i]['source'] = ['- Ако имате машину са CUDA GPU-ом, LLM ће се тренирати на GPU-у без измена у коду:\n', '- Путем `device` подешавања, обезбеђујемо да се подаци учитавају на исту направу као и LLM модел:\n']
            
        # --- 5.2 ---
        elif src.startswith('## 5.2 Training an LLM'):
            nb['cells'][i]['source'] = ['## 5.2 Тренирање LLM-а\n']
        elif src.startswith('- In this section, we finally implement'):
            nb['cells'][i]['source'] = ['- У овом одељку коначно имплементирамо код за тренирање LLM-а:\n', '- Фокусирамо се на једноставну тренинг функцију:\n']
        elif src.startswith('- Now, let us train the LLM'):
            nb['cells'][i]['source'] = ['- Сада тренирајмо LLM користећи горе дефинисану тренинг функцију:\n']
        elif src.startswith('- Looking at the results above'):
            nb['cells'][i]['source'] = ['- Гледајући резултате изнад, видимо да модел почиње са неразумљивим низовима речи, а пред крај производи граматички исправне реченице:\n', '- Међутим, на основу тренинг и валидационих loss-ова, видимо да модел почиње да се претренирава (overfitting):\n', '- Ово се дешава јер имамо веома мали тренинг скуп:\n']
        elif src.startswith('**If you are interested in augmenting**'):
            nb['cells'][i]['source'] = ['**Ако сте заинтересовани за проширење ове тренинг функције напреднијим техникама, погледајте [Додатак D](../../appendix-D/01_main-chapter-code)**\n']
        elif src.startswith('**If you are interested in a larger training**'):
            nb['cells'][i]['source'] = ['**Ако сте заинтересовани за већи тренинг скуп и дуже тренирање, погледајте [../03_bonus_pretraining_on_gutenberg](../03_bonus_pretraining_on_gutenberg)**\n']
            
        # --- 5.3 ---
        elif src.startswith('## 5.3 Decoding strategies to control randomness'):
            nb['cells'][i]['source'] = ['## 5.3 Стратегије декодирања за контролу насумичности\n']
        elif src.startswith('- Inference is relatively cheap'):
            nb['cells'][i]['source'] = ['- Закључивање (inference) је релативно јефтино са малим LLM-ом, тако да нема потребе за GPU-ом:\n']
        elif src.startswith('- Even if we execute the `generate_text_simple`'):
            nb['cells'][i]['source'] = ['- Чак и ако извршимо `generate_text_simple` више пута, LLM ће увек генерисати исте излазе:\n', '- Сада уводимо два концепта, такозване стратегије декодирања: *temperature scaling* и *top-k* семпловање:\n', '- Ово ће омогућити моделу да контролише насумичност и разноликост генерисаног текста:\n']
        elif src.startswith('### 5.3.1 Temperature scaling'):
            nb['cells'][i]['source'] = ['### 5.3.1 Temperature скалирање\n']
        elif src.startswith('- Previously, we always sampled the token'):
            nb['cells'][i]['source'] = ['- Раније смо увек бирали токен са највећом вероватноћом користећи `torch.argmax`:\n', '- Да додамо разноликост, можемо узорковати следећи токен користећи `torch.multinomial(probs, num_samples=1)`:\n']
        elif src.startswith("- Here's a little recap"):
            nb['cells'][i]['source'] = ['- Ево кратког резимеа генерисања следећег токена, претпостављајући веома мали речник:\n']
        elif src.startswith('- Instead of determining the most likely token'):
            nb['cells'][i]['source'] = ['- Уместо одређивања највероватнијег токена путем `torch.argmax`, користимо `torch.multinomial(probas, num_samples=1)`:\n']
        elif src.startswith("- For illustration purposes, let's see"):
            nb['cells'][i]['source'] = ['- За илустрацију, видимо шта се дешава када узоркујемо следећи токен 1.000 пута:\n']
        elif src.startswith('- We can control the distribution'):
            nb['cells'][i]['source'] = ['- Можемо контролисати расподелу кроз концепт temperature скалирања:\n', '- „Temperature scaling“ је дељење logits-а бројем већим од 0:\n', '- Температуре веће од 1 резултирају равномернијом расподелом:\n', '- Температуре мање од 1 резултирају сигурнијом (оштријом) расподелом:\n']
        elif src.startswith('- We can see that the rescaling'):
            nb['cells'][i]['source'] = ['- Видимо да прескалирање температуром 0.1 резултира оштријом расподелом:\n']
        elif src.startswith('- The rescaled probabilities via temperature 5'):
            nb['cells'][i]['source'] = ['- Прескалиране вероватноће температуром 5 су равномерније распоређене:\n']
        elif src.startswith("- Assuming an LLM input"):
            nb['cells'][i]['source'] = ['- Претпостављајући LLM улаз „every effort moves you“, овај приступ може повремено произвести бесмислен текст:\n']
        elif src.startswith('### 5.3.2 Top-k sampling'):
            nb['cells'][i]['source'] = ['### 5.3.2 Top-k семпловање\n']
        elif src.startswith('- To be able to use higher temperatures'):
            nb['cells'][i]['source'] = ['- Да бисмо могли користити више температуре за повећање разноликости излаза и смањење вероватноће бесмислених реченица, ограничавамо узорковане токене на top-k највероватнијих:\n']
        elif src.startswith('- (Please note that the numbers in this figure'):
            nb['cells'][i]['source'] = ['- (Имајте на уму да су бројеви на овој слици скраћени на две децимале; вредности у Softmax реду треба да сумирају на 1.0):\n']
        elif src.startswith('- In code, we can implement this as follows:'):
            nb['cells'][i]['source'] = ['- У коду, ово можемо имплементирати на следећи начин:\n']
        elif src.startswith('> NOTE:'):
            nb['cells'][i]['source'] = ['> НАПОМЕНА:\n']
            
        # --- 5.3.3 ---
        elif src.startswith('### 5.3.3 Modifying the text generation function'):
            nb['cells'][i]['source'] = ['### 5.3.3 Модификација функције за генерисање текста\n']
        elif src.startswith('- The previous two subsections introduced'):
            nb['cells'][i]['source'] = ['- Претходна два пододељка су увела temperature семпловање и top-k семпловање:\n', '- Хајде да користимо ова два концепта да модификујемо `generate_simple` функцију:\n']
            
        # --- 5.4 ---
        elif src.startswith('## 5.4 Loading and saving model weights in PyTorch'):
            nb['cells'][i]['source'] = ['## 5.4 Учитавање и чување тежина модела у PyTorch-у\n']
        elif src.startswith('- Training LLMs is computationally expensive'):
            nb['cells'][i]['source'] = ['- Тренирање LLM-ова је рачунски скупо, тако да је кључно моћи сачувати и учитати тежине:\n']
        elif src.startswith('- The recommended way in PyTorch'):
            nb['cells'][i]['source'] = ['- Препоручени начин у PyTorch-у је чување `state_dict`-а путем `torch.save`:\n']
        elif src.startswith('- Then we can load the model weights'):
            nb['cells'][i]['source'] = ['- Затим можемо учитати тежине у нову `GPTModel` инстанцу:\n']
        elif src.startswith("- It's common to train LLMs with adaptive"):
            nb['cells'][i]['source'] = ['- Уобичајено је тренирати LLM-ове са адаптивним оптимизаторима попут Adam или AdamW:\n', '- Ови оптимизатори чувају додатне параметре за сваку тежину, па их такође чувамо:\n']
            
        # --- 5.5 ---
        elif src.startswith('## 5.5 Loading pretrained weights from OpenAI'):
            nb['cells'][i]['source'] = ['## 5.5 Учитавање предтренираних тежина из OpenAI-ја\n']
        elif src.startswith('- Previously, we only trained a small GPT-2'):
            nb['cells'][i]['source'] = ['- Раније смо тренирали само мали GPT-2 модел на веома малој краткој причи:\n', '- Заинтересовани читаоци могу наћи дуже предтренирање на Project Gutenberg корпусу у [../03_bonus_pretraining_on_gutenberg](../03_bonus_pretraining_on_gutenberg):\n', '- Срећом, не морамо трошити десетине до стотине хиљада долара да предтренирамо модел:\n']
        elif src.startswith('- For an alternative way to load the weights'):
            nb['cells'][i]['source'] = ['- За алтернативни начин учитавања тежина из Hugging Face Hub-а, погледајте [../02_alternative_weight_loading](../02_alternative_weight_loading):\n']
        elif src.startswith('- First, some boilerplate code'):
            nb['cells'][i]['source'] = ['- Прво, код за преузимање датотека од OpenAI-ја и учитавање тежина у Python:\n', '- OpenAI је користио TensorFlow, па ћемо морати инсталирати TensorFlow за учитавање тежина:\n']
        elif src.startswith('- We can then download the model weights'):
            nb['cells'][i]['source'] = ['- Затим можемо преузети тежине модела за 124 милиона параметара:\n']
        elif src.startswith('- Alternatively, "355M"'):
            nb['cells'][i]['source'] = ['- Алтернативно, "355M", "774M" и "1558M" су такође подржани:\n', '- Разлика између ових модела различитих величина сумирана је на слици испод:\n']
        elif src.startswith('- Above, we loaded the 124M GPT-2'):
            nb['cells'][i]['source'] = ['- Горе смо учитали 124M GPT-2 тежине у Python, али их још треба пренети у нашу `GPTModel` инстанцу:\n', '- Прво иницијализујемо нову GPTModel инстанцу:\n', '- Оригинални GPT модел је иницијализовао linear слојеве са bias векторима; да бисмо исправно учитали тежине, морамо омогућити `qkv_bias=True`:\n', '- Такође користимо дужину контекста од 1024 токена:\n']
        elif src.startswith('- The next task is to assign'):
            nb['cells'][i]['source'] = ['- Следећи задатак је доделити OpenAI тежине одговарајућим тензорима тежина у нашој `GPTModel` инстанци:\n']
        elif src.startswith('- If the model is loaded correctly'):
            nb['cells'][i]['source'] = ['- Ако је модел исправно учитан, можемо га користити за генерисање новог текста:\n']
        elif src.startswith('- We know that we loaded the model weights correctly'):
            nb['cells'][i]['source'] = ['- Знамо да смо исправно учитали тежине јер модел може да генерише кохерентан текст:\n']
            
        # --- Summary ---
        elif src.startswith('## Summary and takeaways'):
            nb['cells'][i]['source'] = ['## Резиме и закључци\n']
        elif src.startswith('- See the [./gpt_train.py]'):
            nb['cells'][i]['source'] = ['- Погледајте скрипту [./gpt_train.py](./gpt_train.py) за самостално тренирање.\n', '- Скрипта [./gpt_generate.py](./gpt_generate.py) учитава предтрениране тежине и генерише текст.\n', '- Решења вежби можете наћи у [./exercise-solutions.ipynb](./exercise-solutions.ipynb).\n']
            
        # Additional specific cells
        elif src.startswith('# Uncomment the following'):
            nb['cells'][i]['source'] = ['# Откоментаришите следећи код за израчунавање времена извршавања:\n']

with open('ch05.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Превод ch05.ipynb је завршен!')
