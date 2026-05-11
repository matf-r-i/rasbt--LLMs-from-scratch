import json

with open('ch04.ipynb', 'r', encoding='utf-8') as f:
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
        elif src.startswith('# Chapter 4: Implementing a GPT model from Scratch To Generate Text'):
            nb['cells'][i]['source'] = ['# Поглавље 4: Имплементација GPT модела од нуле за генерисање текста\n']
        elif src.startswith('Packages that are being used') or src.startswith('from importlib.metadata'):
            nb['cells'][i]['source'] = ['Пакети који се користе у овој Jupyter свесци:\n']
        elif src.startswith('- In this chapter, we implement a GPT-like'):
            nb['cells'][i]['source'] = ['- У овом поглављу имплементирамо GPT архитектуру; следеће поглавље ће се фокусирати на тренирање овог LLM-а:\n']
            
        # --- 4.1 ---
        elif src.startswith('## 4.1 Coding an LLM architecture'):
            nb['cells'][i]['source'] = ['## 4.1 Програмирање LLM архитектуре\n']
        elif src.startswith('- Chapter 1 discussed models like GPT and Llama'):
            nb['cells'][i]['source'] = ['- Поглавље 1 је говорило о моделима попут GPT и Llama, који генеришу речи секвенцијално и засновани су на decoder делу оригиналне transformer архитектуре:\n', '- Стога се ови LLM-ови често називају „decoder-like“ LLM-овима\n', '- LLM-ови су већи од конвенционалних модела дубоког учења, углавном због великог броја параметара:\n']
        elif src.startswith('- In previous chapters, we used small embedding'):
            nb['cells'][i]['source'] = ['- У претходним поглављима користили смо мале embedding димензије ради илустрације:\n', '- У овом поглављу користимо величине модела сличне малом GPT-2 моделу\n', '- Кодираћемо архитектуру најмањег GPT-2 модела (124 милиона параметара):\n']
        elif src.startswith('- Configuration details for the 124 million'):
            nb['cells'][i]['source'] = ['- Детаљи конфигурације за GPT-2 модел од 124 милиона параметара:\n']
        elif src.startswith('- We use short variable names'):
            nb['cells'][i]['source'] = ['- Користимо кратке називе променљивих да избегнемо дугачке линије кода:\n', '- `"vocab_size"` означава величину речника од 50.257 речи\n', '- `"context_length"` представља максимални број улазних токена\n', '- `"emb_dim"` је embedding величина за улазне токене (768)\n', '- `"n_heads"` је број глава пажње у multi-head attention механизму\n', '- `"n_layers"` је број transformer блокова у моделу\n', '- `"drop_rate"` је интензитет dropout механизма (0.1 = 10%)\n', '- `"qkv_bias"` одлучује да ли Linear слојеви треба да садрже bias вектор\n']

        # --- 4.2 ---
        elif src.startswith('## 4.2 Normalizing activations with layer normalization'):
            nb['cells'][i]['source'] = ['## 4.2 Нормализација активација слојном нормализацијом (LayerNorm)\n']
        elif src.startswith('- Layer normalization, also known as LayerNorm'):
            nb['cells'][i]['source'] = ['- LayerNorm центрира активације слоја око средње вредности 0 и нормализује варијансу на 1:\n', '- Ово стабилизује тренирање и омогућава бржу конвергенцију\n', '- LayerNorm се примењује пре и после multi-head attention модула:\n']
        elif src.startswith("- Let's see how layer normalization works"):
            nb['cells'][i]['source'] = ['- Да видимо како LayerNorm ради пропуштањем малог улазног узорка кроз једноставан неуронски слој:\n']
        elif src.startswith("- Let's compute the mean"):
            nb['cells'][i]['source'] = ['- Израчунајмо средњу вредност и варијансу за сваки од 2 улаза горе:\n']
        elif src.startswith('- The normalization is applied'):
            nb['cells'][i]['source'] = ['- Нормализација се примењује на сваки од два улаза (редова) независно:\n']
        elif src.startswith('- Subtracting the mean and dividing'):
            nb['cells'][i]['source'] = ['- Одузимање средње вредности и дељење квадратним кореном варијансе центрира улазе на средњу вредност 0 и варијансу 1:\n']
        elif src.startswith('- Each input is centered at 0'):
            nb['cells'][i]['source'] = ['- Сваки улаз је центриран на 0 и има јединичну варијансу 1:\n']
        elif src.startswith('- Above, we normalized the features'):
            nb['cells'][i]['source'] = ['- Горе смо нормализовали особине сваког улаза:\n', '- Сада, користећи исту идеју, можемо имплементирати `LayerNorm` класу:\n']
        elif src.startswith('**Scale and shift**'):
            nb['cells'][i]['source'] = ['**Scale и Shift параметри**\n']
        elif src.startswith('- Note that in addition to performing the normalization'):
            nb['cells'][i]['source'] = ['- Поред нормализације, додали смо два тренирајућа параметра, `scale` и `shift`:\n', '- Почетне вредности (1 и 0) немају ефекат, али су тренирајући параметри које LLM прилагођава током тренирања:\n']
        elif src.startswith('**Biased variance**'):
            nb['cells'][i]['source'] = ['**Пристрасна варијанса (Biased variance)**\n']
        elif src.startswith('- In the variance calculation above'):
            nb['cells'][i]['source'] = ['- У израчунавању варијансе изнад, `unbiased=False` користи формулу без Беселове корекције:\n', '- За LLM-ове, где је embedding димензија веома велика, разлика је занемарљива:\n', '- GPT-2 је трениран са пристрасном варијансом, због чега смо усвојили ово подешавање:\n']
        elif src.startswith("- Let's now try out `LayerNorm`"):
            nb['cells'][i]['source'] = ['- Пробајмо сада `LayerNorm` у пракси:\n']
            
        # --- 4.3 ---
        elif src.startswith('## 4.3 Implementing a feed forward network with GELU activations'):
            nb['cells'][i]['source'] = ['## 4.3 Имплементација feed forward мреже са GELU активацијама\n']
        elif src.startswith('- In this section, we implement a small'):
            nb['cells'][i]['source'] = ['- У овом одељку имплементирамо мали неуронски подмодул који се користи као део transformer блока:\n']
        elif src.startswith('- We start with the activation function'):
            nb['cells'][i]['source'] = ['- Почињемо са активационом функцијом:\n', '- ReLU се обично користи у дубоком учењу\n', '- У LLM-овима се користе GELU и SwiGLU активације\n']
        elif src.startswith('- GELU ([Hendrycks and Gimpel 2016]'):
            nb['cells'][i]['source'] = ['- GELU се може имплементирати на неколико начина; апроксимација која се користи у оригиналном GPT-2:\n']
        elif src.startswith('- As we can see, ReLU is a piecewise'):
            nb['cells'][i]['source'] = ['- Као што видимо, ReLU је линеарна функција по деловима:\n', '- GELU је глатка нелинеарна функција која апроксимира ReLU са не-нултим градијентом за негативне вредности:\n']
        elif src.startswith('- Next, let us implement the small'):
            nb['cells'][i]['source'] = ['- Затим, имплементирајмо `FeedForward` модул који ћемо користити у transformer блоку:\n']
            
        # --- 4.4 ---
        elif src.startswith('## 4.4 Adding shortcut connections'):
            nb['cells'][i]['source'] = ['## 4.4 Додавање shortcut (пречица) веза\n']
        elif src.startswith("- Next, let's talk about the concept"):
            nb['cells'][i]['source'] = ['- Shortcut везе, такође познате као skip или residual везе:\n', '- Оригинално предложене у дубоким мрежама за компјутерски вид ради ублажавања проблема нестајућих градијената:\n', '- Shortcut веза ствара алтернативну краћу путању за проток градијента кроз мрежу:\n', '- Ово се постиже додавањем излаза једног слоја излазу каснијег слоја:\n']
        elif src.startswith('- In code, it looks like this:'):
            nb['cells'][i]['source'] = ['- У коду, то изгледа овако:\n']
        elif src.startswith("- Let's print the gradient values first **without**"):
            nb['cells'][i]['source'] = ['- Штампајмо вредности градијената прво **без** shortcut веза:\n']
        elif src.startswith("- Next, let's print the gradient values **with**"):
            nb['cells'][i]['source'] = ['- Затим, штампајмо вредности градијената **са** shortcut везама:\n']
        elif src.startswith('- As we can see based on the output above'):
            nb['cells'][i]['source'] = ['- Као што видимо, shortcut везе спречавају нестајање градијената у раним слојевима:\n']
            
        # --- 4.5 ---
        elif src.startswith('## 4.5 Connecting attention and linear layers in a transformer block'):
            nb['cells'][i]['source'] = ['## 4.5 Повезивање attention и linear слојева у transformer блоку\n']
        elif src.startswith('- In this section, we now combine'):
            nb['cells'][i]['source'] = ['- У овом одељку комбинујемо претходне концепте у transformer блок:\n', '- Transformer блок комбинује causal multi-head attention (из претходног поглавља) са feed forward мрежом коју смо имплементирали раније:\n', '- Такође користи dropout и shortcut везе:\n']
        elif src.startswith('- Suppose we have 2 input samples'):
            nb['cells'][i]['source'] = ['- Претпоставимо 2 улазна узорка са по 6 токена, сваки токен је 768-димензионални вектор:\n', '- Transformer блок примењује self-attention, затим linear слојеве, и производи излаз сличне величине:\n']
            
        # --- 4.6 ---
        elif src.startswith('## 4.6 Coding the GPT model'):
            nb['cells'][i]['source'] = ['## 4.6 Програмирање GPT модела\n']
        elif src.startswith('- We are almost there: now let us plug'):
            nb['cells'][i]['source'] = ['- Скоро смо готови: убацимо transformer блок у архитектуру коју смо кодирали на почетку поглавља:\n', '- Transformer блок се понавља више пута; за најмањи 124M GPT-2 модел, понављамо га 12 пута:\n']
        elif src.startswith('- The corresponding code implementation'):
            nb['cells'][i]['source'] = ['- Одговарајућа имплементација кода, где `cfg["n_layers"] = 12`:\n']
        elif src.startswith('- Using the configuration of the 124M'):
            nb['cells'][i]['source'] = ['- Користећи конфигурацију 124M параметарског модела, можемо инстанцирати GPT модел са насумичним почетним тежинама:\n']
        elif src.startswith('- We will train this model in the next chapter'):
            nb['cells'][i]['source'] = ['- Тренираћемо овај модел у следећем поглављу:\n', '- Брза напомена о величини: раније смо га називали 124M параметарским моделом; проверимо овај број:\n']
        elif src.startswith('- As we see above, this model has 163M'):
            nb['cells'][i]['source'] = ['- Као што видимо, модел има 163M, а не 124M параметара. Зашто?\n', '- У оригиналном GPT-2 раду, истраживачи су применили weight tying (дељење тежина)\n', '- Поново су користили embedding слој токена (`tok_emb`) као излазни слој\n', '- Ако одузмемо број параметара излазног слоја, добијамо 124M параметарски модел:\n']
        elif src.startswith('- In practice, I found it easier to train'):
            nb['cells'][i]['source'] = ['- У пракси је лакше тренирати модел без weight-tying-а, због чега га нисмо имплементирали овде:\n', '- Вратићемо се на ову идеју када будемо учитавали претходно трениране тежине у поглављу 5:\n']
        elif src.startswith('- Lastly, we can compute the memory requirements'):
            nb['cells'][i]['source'] = ['- Коначно, можемо израчунати меморијске захтеве модела:\n']
        elif src.startswith('- Exercise: you can try the following'):
            nb['cells'][i]['source'] = ['- Вежба: можете испробати следеће конфигурације из GPT-2 рада:\n', '    - **GPT2-small**: emb_dim=768, n_layers=12, n_heads=12\n', '    - **GPT2-medium**: emb_dim=1024, n_layers=24, n_heads=16\n', '    - **GPT2-large**: emb_dim=1280, n_layers=36, n_heads=20\n', '    - **GPT2-XL**: emb_dim=1600, n_layers=48, n_heads=25\n']
            
        # --- 4.7 ---
        elif src.startswith('## 4.7 Generating text'):
            nb['cells'][i]['source'] = ['## 4.7 Генерисање текста\n']
        elif src.startswith('- LLMs like the GPT model we implemented'):
            nb['cells'][i]['source'] = ['- LLM-ови попут GPT модела који смо имплементирали генеришу једну реч по једну:\n']
        elif src.startswith('- The following `generate_text_simple` function'):
            nb['cells'][i]['source'] = ['- Функција `generate_text_simple` имплементира похлепно декодирање (greedy decoding):\n', '- У похлепном декодирању, модел бира реч са највећом вероватноћом као следећи излаз:\n', '- У следећем поглављу ћемо имплементирати напреднију `generate_text` функцију:\n']
        elif src.startswith('- The `generate_text_simple` above implements'):
            nb['cells'][i]['source'] = ['- `generate_text_simple` изнад имплементира итеративни процес, где креира један токен по један:\n']
        elif src.startswith("- Let's prepare an input example:"):
            nb['cells'][i]['source'] = ['- Припремимо улазни пример:\n']
        elif src.startswith('- Remove batch dimension and convert back'):
            nb['cells'][i]['source'] = ['- Уклонимо batch димензију и конвертујмо назад у текст:\n']
        elif src.startswith('- Note that the model is untrained'):
            nb['cells'][i]['source'] = ['- Имајте на уму да модел није трениран; отуда насумични излазни текстови:\n', '- Тренираћемо модел у следећем поглављу:\n']
            
        # --- Summary ---
        elif src.startswith('## Summary and takeaways'):
            nb['cells'][i]['source'] = ['## Резиме и закључци\n']
        elif src.startswith('- See the [./gpt.py]'):
            nb['cells'][i]['source'] = ['- Погледајте скрипту [./gpt.py](./gpt.py) која садржи GPT модел имплементиран у овој Jupyter свесци.\n', '- Решења вежби можете наћи у [./exercise-solutions.ipynb](./exercise-solutions.ipynb).\n']
            
        # Remaining specific cells
        elif src.startswith('---'):
            nb['cells'][i]['source'] = ['---\n']
        elif src.startswith('**Note**'):
            nb['cells'][i]['source'] = ['**Напомена**\n']
        elif src.startswith('- If you are running this code on Windows'):
            nb['cells'][i]['source'] = ['- Ако покрећете овај код на Windows-у или Linux-у, горње вредности могу изгледати овако:\n']
        elif src.startswith('- Since these are just random numbers'):
            nb['cells'][i]['source'] = ['- Пошто су ово само насумични бројеви, то није разлог за забринутост:\n']

with open('ch04.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Превод ch04.ipynb је завршен!')
