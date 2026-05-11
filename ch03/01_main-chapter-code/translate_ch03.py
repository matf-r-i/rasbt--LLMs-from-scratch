import json

with open('ch03.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source']).strip()
        
        # --- Header/Intro ---
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
        elif src.startswith('# Chapter 3: Coding Attention Mechanisms'):
            nb['cells'][i]['source'] = ['# Поглавље 3: Програмирање механизама пажње (Attention Mechanisms)\n']
        elif src.startswith('Packages that are being used in this notebook:'):
            nb['cells'][i]['source'] = ['Пакети који се користе у овој Jupyter свесци:\n']
        elif src.startswith('- This chapter covers attention mechanisms, the engine of LLMs'):
            nb['cells'][i]['source'] = ['- Ово поглавље покрива механизме пажње (attention mechanisms), покретачку снагу LLM-ова:\n']
            
        # --- 3.1 ---
        elif src.startswith('## 3.1 The problem with modeling long sequences'):
            nb['cells'][i]['source'] = ['## 3.1 Проблем моделирања дугих секвенци\n']
        elif src.startswith('- No code in this section') and '3.1' in ''.join(nb['cells'][i]['source']):
            nb['cells'][i]['source'] = ['- Нема кода у овом одељку\n']
        elif src.startswith('- Translating a text word by word'):
            nb['cells'][i]['source'] = ['- Превођење текста реч по реч није изводљиво због разлика у граматичким структурама између изворног и циљног језика:\n']
        elif src.startswith('- Prior to the introduction of transformer models'):
            nb['cells'][i]['source'] = ['- Пре увођења transformer модела, encoder-decoder RNN-ови су се обично користили за задатке машинског превођења\n', '- У овој поставци, encoder обрађује секвенцу токена из изворног језика, користећи скривено стање (hidden state) за генерисање сажете репрезентације целе улазне секвенце:\n']
            
        # --- 3.2 ---
        elif src.startswith('## 3.2 Capturing data dependencies with attention mechanisms'):
            nb['cells'][i]['source'] = ['## 3.2 Хватање зависности података помоћу механизама пажње\n']
        elif src.startswith('- Through an attention mechanism'):
            nb['cells'][i]['source'] = ['- Кроз механизам пажње, decoder сегмент мреже који генерише текст може селективно приступити свим улазним токенима:\n']
        elif src.startswith('- Self-attention in transformers is a technique'):
            nb['cells'][i]['source'] = ['- Самопажња (self-attention) у transformers-има је техника дизајнирана да побољша репрезентације улаза омогућавајући свакој позицији у секвенци да интерагује са сваком другом позицијом у истој секвенци:\n']
            
        # --- 3.3 ---
        elif src.startswith('## 3.3 Attending to different parts of the input with self-attention'):
            nb['cells'][i]['source'] = ['## 3.3 Обраћање пажње на различите делове улаза помоћу самопажње\n']
        elif src.startswith('### 3.3.1 A simple self-attention mechanism without trainable weights'):
            nb['cells'][i]['source'] = ['### 3.3.1 Једноставан механизам самопажње без тежина које се могу тренирати\n']
        elif src.startswith('- This section explains a very simplified variant'):
            nb['cells'][i]['source'] = ['- Овај одељак објашњава веома поједностављену варијанту самопажње, која не садржи тежине које се могу тренирати\n', '- Ово је само за илустрацију и НИЈЕ механизам пажње који се користи у transformers-има\n', '- Следећи одељак, 3.3.2, ће проширити овај једноставни механизам да имплементира прави механизам самопажње\n']
        elif src.startswith('- Suppose we are given an input sequence'):
            nb['cells'][i]['source'] = ['- Претпоставимо да нам је дата улазна секвенца $x^{(1)}$ до $x^{(T)}$:\n', '  - Улаз је текст који је претворен у векторске репрезентације токена (token embeddings) како је описано у поглављу 2\n', '  - На пример, $x^{(1)}$ је d-димензионални вектор који представља реч „Your“, итд.\n']
        elif src.startswith('- **Goal:** compute context vectors'):
            nb['cells'][i]['source'] = ['- **Циљ:** израчунати контекстне векторе $z^{(i)}$ за сваки елемент улазне секвенце $x^{(i)}$ у $x^{(1)}$ до $x^{(T)}$ (где $z$ и $x$ имају исту димензију):\n', '    - Контекстни вектор $z^{(i)}$ је пондерисана сума над улазима $x^{(1)}$ до $x^{(T)}$\n', '    - Контекстни вектор је специфичан за одређени улаз\n']
        elif src.startswith('- (Please note that the numbers'):
            nb['cells'][i]['source'] = ['- (Имајте на уму да су бројеви на овој слици скраћени на једну децималу ради смањења визуелне збрке; друге слике такође могу садржати скраћене вредности)\n']
        elif src.startswith('- By convention, the unnormalized attention weights'):
            nb['cells'][i]['source'] = ['- По конвенцији, ненормализоване тежине пажње називају се **„attention scores“**, док се нормализоване тежине пажње, које сумирају на 1, називају **„attention weights“**\n']
        elif src.startswith('- The code below walks through the figure above step by step'):
            nb['cells'][i]['source'] = ['- Код испод пролази кроз горњу слику корак по корак\n']
        elif src.startswith('- **Step 1:** compute unnormalized attention scores'):
            nb['cells'][i]['source'] = ['- **Корак 1:** израчунати ненормализоване attention scores $\\omega$\n', '- Претпоставимо да користимо други улазни токен као упит (query), тј. $q^{(2)} = x^{(2)}$, рачунамо ненормализоване резултате пажње путем скаларних производа (dot products):\n', '    - $\\omega_{21} = x^{(1)} q^{(2)\\top}$\n', '    - $\\omega_{22} = x^{(2)} q^{(2)\\top}$\n', '    - $\\omega_{23} = x^{(3)} q^{(2)\\top}$\n', '    - ...\n', '    - $\\omega_{2T} = x^{(T)} q^{(2)\\top}$\n']
        elif src.startswith('- Suppose we have the following input sentence'):
            nb['cells'][i]['source'] = ['- Претпоставимо да имамо следећу улазну реченицу која је већ уграђена у 3-димензионалне векторе (користимо малу димензију за илустрацију):\n']
        elif src.startswith('- (In this book, we follow'):
            nb['cells'][i]['source'] = ['- (У овој књизи пратимо конвенцију машинског учења где су примери тренирања представљени као редови, а вредности особина као колоне)\n']
        elif src.startswith('- The primary objective of this section'):
            nb['cells'][i]['source'] = ['- Примарни циљ овог одељка је да демонстрира како се контекстни вектор $z^{(2)}$ израчунава користећи другу улазну секвенцу $x^{(2)}$ као упит (query)\n']
        elif src.startswith('- The figure depicts the initial step'):
            nb['cells'][i]['source'] = ['- Слика приказује почетни корак, који укључује израчунавање attention scores ω између $x^{(2)}$ и свих других улазних елемената:\n']
        elif src.startswith('- We use input sequence element 2'):
            nb['cells'][i]['source'] = ['- Користимо елемент 2 улазне секвенце, $x^{(2)}$, као пример за израчунавање контекстног вектора $z^{(2)}$:\n', '- Први корак је израчунавање ненормализованих attention scores:\n']
        elif src.startswith('- Side note: a dot product is essentially'):
            nb['cells'][i]['source'] = ['- Напомена: скаларни производ (dot product) је скраћеница за множење два вектора елемент по елемент и сабирање резултата:\n']
        elif src.startswith('- **Step 2:** normalize the unnormalized attention scores'):
            nb['cells'][i]['source'] = ['- **Корак 2:** нормализовати ненормализоване attention scores („omega“, $\\omega$) тако да сумирају на 1:\n', '- Једноставан начин да се нормализују:\n']
        elif src.startswith('- However, in practice, using the softmax function'):
            nb['cells'][i]['source'] = ['- Међутим, у пракси се препоручује коришћење softmax функције за нормализацију, која боље рукује екстремним вредностима и има пожељније градијентне особине током тренирања:\n']
        elif src.startswith('- The naive implementation above can suffer'):
            nb['cells'][i]['source'] = ['- Наивна имплементација изнад може патити од нумеричке нестабилности за велике или мале улазне вредности:\n', '- Стога се препоручује коришћење PyTorch имплементације softmax-а:\n']
        elif src.startswith('- **Step 3**: compute the context vector'):
            nb['cells'][i]['source'] = ['- **Корак 3:** израчунати контекстни вектор $z^{(2)}$ множењем уграђених улазних токена $x^{(i)}$ са attention weights и сабирањем резултујућих вектора:\n']
            
        # --- 3.3.2 ---
        elif src.startswith('### 3.3.2 Computing attention weights for all input tokens'):
            nb['cells'][i]['source'] = ['### 3.3.2 Израчунавање attention weights за све улазне токене\n']
        elif src.startswith('#### Generalize to all input sequence tokens'):
            nb['cells'][i]['source'] = ['#### Генерализација на све токене улазне секвенце:\n']
        elif src.startswith('- Above, we computed the attention weights'):
            nb['cells'][i]['source'] = ['- Горе смо израчунали attention weights и контекстни вектор за улаз 2:\n', '- Сада генерализујемо ово израчунавање на све attention weights и контекстне векторе:\n']
        elif src.startswith('- (Please note that the numbers in this figure are truncated to two'):
            nb['cells'][i]['source'] = ['- (Имајте на уму да су бројеви скраћени на две децимале; вредности у сваком реду треба да сумирају на 1.0)\n']
        elif src.startswith('- In self-attention, the process starts'):
            nb['cells'][i]['source'] = ['- У самопажњи, процес почиње израчунавањем attention scores, који се затим нормализују да би се добили attention weights који сумирају на 1:\n']
        elif src.startswith('- Apply previous **step 1**'):
            nb['cells'][i]['source'] = ['- Применити претходни **корак 1** на све парове елемената да би се израчунала матрица ненормализованих attention scores:\n']
        elif src.startswith('- We can achieve the same as above more efficiently'):
            nb['cells'][i]['source'] = ['- Исто можемо постићи ефикасније путем матричног множења:\n']
        elif src.startswith('- Similar to **step 2** previously'):
            nb['cells'][i]['source'] = ['- Слично **кораку 2**, нормализујемо сваки ред тако да вредности у сваком реду сумирају на 1:\n']
        elif src.startswith('- Quick verification'):
            nb['cells'][i]['source'] = ['- Брза провера да вредности у сваком реду заиста сумирају на 1:\n']
        elif src.startswith('- Apply previous **step 3**'):
            nb['cells'][i]['source'] = ['- Применити претходни **корак 3** да се израчунају сви контекстни вектори:\n']
        elif src.startswith('- As a sanity check'):
            nb['cells'][i]['source'] = ['- Као провера, претходно израчунати контекстни вектор $z^{(2)}$ може се наћи у 2. реду горе:\n']
            
        # --- 3.4 ---
        elif src.startswith('## 3.4 Implementing self-attention with trainable weights'):
            nb['cells'][i]['source'] = ['## 3.4 Имплементација самопажње са тежинама које се могу тренирати\n']
        elif src.startswith('- A conceptual framework illustrating'):
            nb['cells'][i]['source'] = ['- Концептуални оквир који илуструје како се механизам самопажње уклапа у причу и структуру ове књиге и поглавља:\n']
        elif src.startswith('### 3.4.1 Computing the attention weights step by step'):
            nb['cells'][i]['source'] = ['### 3.4.1 Израчунавање attention weights корак по корак\n']
        elif src.startswith('- In this section, we are implementing'):
            nb['cells'][i]['source'] = ['- У овом одељку имплементирамо механизам самопажње који се користи у оригиналној transformer архитектури, GPT моделима и већини других популарних LLM-ова:\n', '- Овај механизам се такође назива „scaled dot-product attention“\n']
        elif src.startswith('- The overall idea is similar to before'):
            nb['cells'][i]['source'] = ['- Основна идеја је слична раније:\n', '  - Желимо да израчунамо контекстне векторе као пондерисане суме над улазним векторима\n', '  - За то су нам потребни attention weights\n']
        elif src.startswith('- As you will see, there are only slight differences'):
            nb['cells'][i]['source'] = ['- Као што ћете видети, постоје само мале разлике у односу на основни механизам пажње:\n', '  - Најзначајнија разлика је увођење матрица тежина које се ажурирају током тренирања модела\n', '  - Ове матрице су кључне да модел научи да производи „добре“ контекстне векторе:\n']
        elif src.startswith('- Implementing the self-attention mechanism step by step'):
            nb['cells'][i]['source'] = ['- Имплементирамо механизам самопажње корак по корак, уводећи три матрице тежина $W_q$, $W_k$ и $W_v$:\n', '- Ове три матрице се користе за пројектовање уграђених улазних токена $x^{(i)}$ у query, key и value векторе:\n', '  - Query вектор: $q^{(i)} = W_q x^{(i)}$\n', '  - Key вектор: $k^{(i)} = W_k x^{(i)}$\n', '  - Value вектор: $v^{(i)} = W_v x^{(i)}$\n']
        elif src.startswith('- The embedding dimensions of the input'):
            nb['cells'][i]['source'] = ['- Димензије embedding-а улаза $x$ и query вектора $q$ могу бити исте или различите:\n', '- У GPT моделима, улазне и излазне димензије су обично исте, али за илустрацију бирамо различите димензије:\n']
        elif src.startswith('- Below, we initialize the three weight matrices'):
            nb['cells'][i]['source'] = ['- Испод иницијализујемо три матрице тежина; подешавамо `requires_grad=False` ради смањења визуелне збрке:\n']
        elif src.startswith('- Next we compute the query, key, and value vectors'):
            nb['cells'][i]['source'] = ['- Затим израчунавамо query, key и value векторе:\n']
        elif src.startswith('- As we can see below'):
            nb['cells'][i]['source'] = ['- Као што видимо испод, успешно смо пројектовали 6 улазних токена из 3D у 2D embedding простор:\n']
        elif src.startswith('- In the next step, **step 2**'):
            nb['cells'][i]['source'] = ['- У следећем **кораку 2**, израчунавамо ненормализоване attention scores скаларним производом између query и сваког key вектора:\n']
        elif src.startswith('- Since we have 6 inputs'):
            nb['cells'][i]['source'] = ['- Пошто имамо 6 улаза, имамо 6 attention scores за дати query вектор:\n']
        elif src.startswith('- Next, in **step 3**'):
            nb['cells'][i]['source'] = ['- Затим, у **кораку 3**, израчунавамо attention weights користећи softmax функцију:\n', '- Разлика је у томе што сада скалирамо attention scores дељењем са квадратним кореном димензије embedding-а, $\\sqrt{d_k}$:\n']
        elif src.startswith('- In **step 4**'):
            nb['cells'][i]['source'] = ['- У **кораку 4**, израчунавамо контекстни вектор за улазни query вектор 2:\n']
        elif src.startswith('### 3.4.2 Implementing a compact SelfAttention class'):
            nb['cells'][i]['source'] = ['### 3.4.2 Имплементација компактне SelfAttention класе\n']
        elif src.startswith('- Putting it all together'):
            nb['cells'][i]['source'] = ['- Спајајући све заједно, можемо имплементирати механизам самопажње на следећи начин:\n']
        elif src.startswith('- We can streamline the implementation'):
            nb['cells'][i]['source'] = ['- Можемо побољшати имплементацију користећи PyTorch Linear слојеве:\n', '- Предност `nn.Linear`-а је боља иницијализација тежина:\n']
        elif src.startswith('- Note that `SelfAttention_v1`'):
            nb['cells'][i]['source'] = ['- Имајте на уму да `SelfAttention_v1` и `SelfAttention_v2` дају различите излазе јер користе различите почетне тежине:\n']
            
        # --- 3.5 ---
        elif src.startswith('## 3.5 Hiding future words with causal attention'):
            nb['cells'][i]['source'] = ['## 3.5 Скривање будућих речи узрочном пажњом (causal attention)\n']
        elif src.startswith('- In causal attention'):
            nb['cells'][i]['source'] = ['- У узрочној пажњи (causal attention), attention weights изнад дијагонале су маскирани, спречавајући LLM да користи будуће токене при израчунавању контекстних вектора:\n']
        elif src.startswith('### 3.5.1 Applying a causal attention mask'):
            nb['cells'][i]['source'] = ['### 3.5.1 Примена маске узрочне пажње\n']
        elif src.startswith('- In this section, we are converting'):
            nb['cells'][i]['source'] = ['- У овом одељку конвертујемо претходни механизам самопажње у узрочну самопажњу (causal self-attention):\n', '- Causal self-attention обезбеђује да предвиђање модела за одређену позицију зависи само од претходних позиција:\n', '- Да бисмо то постигли, маскирамо будуће токене:\n']
        elif src.startswith('- To illustrate and implement'):
            nb['cells'][i]['source'] = ['- Да илуструјемо и имплементирамо causal self-attention, радимо са attention scores и weights из претходног одељка:\n']
        elif src.startswith('- The simplest way to mask out future attention weights'):
            nb['cells'][i]['source'] = ['- Најједноставнији начин да маскирамо будуће attention weights је креирањем маске помоћу PyTorch tril функције:\n']
        elif src.startswith('- Then, we can multiply the attention weights'):
            nb['cells'][i]['source'] = ['- Затим можемо помножити attention weights са овом маском да поништимо вредности изнад дијагонале:\n']
        elif src.startswith('- However, if the mask were applied after softmax'):
            nb['cells'][i]['source'] = ['- Међутим, ако се маска примени након softmax-а, то би пореметило расподелу вероватноће коју softmax ствара:\n']
        elif src.startswith('- To make sure that the rows sum to 1'):
            nb['cells'][i]['source'] = ['- Да бисмо осигурали да редови сумирају на 1, можемо нормализовати attention weights:\n']
        elif src.startswith('- While we are technically done'):
            nb['cells'][i]['source'] = ['- Иако смо технички завршили, погледајмо ефикаснији приступ:\n', '- Уместо поништавања вредности и ренормализације, маскирамо ненормализоване attention scores са негативном бесконачношћу пре softmax функције:\n']
        elif src.startswith('- As we can see below'):
            nb['cells'][i]['source'] = ['- Као што видимо испод, сада attention weights у сваком реду поново исправно сумирају на 1:\n']
        elif src.startswith('### 3.5.2 Masking additional attention weights with dropout'):
            nb['cells'][i]['source'] = ['### 3.5.2 Додатно маскирање attention weights dropout-ом\n']
        elif src.startswith('- In addition, we also apply dropout'):
            nb['cells'][i]['source'] = ['- Поред тога, примењујемо dropout да смањимо претренирање (overfitting) током тренирања:\n', '- Dropout се може применити на више места:\n', '  - након израчунавања attention weights;\n', '  - или након множења attention weights са value векторима\n']
        elif src.startswith('- Furthermore, in this specific example'):
            nb['cells'][i]['source'] = ['- У овом примеру користимо dropout стопу од 50% (касније ћемо користити ниже стопе као 0.1 или 0.2):\n']
        elif src.startswith('- If we apply a dropout rate of 0.5'):
            nb['cells'][i]['source'] = ['- Ако применимо dropout стопу од 0.5 (50%), не-избачене вредности ће бити скалиране са фактором 1/0.5 = 2:\n']
        elif src.startswith('- Note that the resulting dropout outputs'):
            nb['cells'][i]['source'] = ['- Имајте на уму да резултујући dropout излази могу изгледати другачије у зависности од вашег оперативног система:\n']
        elif src.startswith('### 3.5.3 Implementing a compact causal self-attention class'):
            nb['cells'][i]['source'] = ['### 3.5.3 Имплементација компактне класе за узрочну самопажњу\n']
        elif src.startswith('- Now, we are ready to implement'):
            nb['cells'][i]['source'] = ['- Сада смо спремни да имплементирамо потпуну имплементацију самопажње, укључујући causal и dropout маске:\n', '- Такође имплементирамо подршку за пакете (batches) са више од једног улаза:\n']
        elif src.startswith('- For simplicity, to simulate such batch input'):
            nb['cells'][i]['source'] = ['- Ради једноставности, дуплирамо улазни текст да симулирамо пакетни улаз:\n']
        elif src.startswith('- Note that dropout is only applied during training'):
            nb['cells'][i]['source'] = ['- Имајте на уму да се dropout примењује само током тренирања, не током закључивања (inference):\n']
            
        # --- 3.6 ---
        elif src.startswith('## 3.6 Extending single-head attention to multi-head attention'):
            nb['cells'][i]['source'] = ['## 3.6 Проширење једно-главе пажње на више-главу пажњу (multi-head attention)\n']
        elif src.startswith('### 3.6.1 Stacking multiple single-head attention layers'):
            nb['cells'][i]['source'] = ['### 3.6.1 Слагање више једно-главих слојева пажње\n']
        elif src.startswith('- Below is a summary'):
            nb['cells'][i]['source'] = ['- Испод је резиме самопажње имплементиране раније (causal и dropout маске нису приказане ради једноставности):\n', '- Ово се назива и једно-глава пажња (single-head attention):\n']
        elif src.startswith('- We simply stack multiple single-head attention modules'):
            nb['cells'][i]['source'] = ['- Једноставно слажемо више једно-главих модула пажње да добијемо модул са више глава:\n']
        elif src.startswith('- The main idea behind multi-head attention'):
            nb['cells'][i]['source'] = ['- Главна идеја иза multi-head attention је да покренемо механизам пажње више пута (паралелно) са различитим, наученим линеарним пројекцијама:\n']
        elif src.startswith('- In the implementation above'):
            nb['cells'][i]['source'] = ['- У имплементацији изнад, димензија embedding-а је 4, јер имамо `d_out=2` и 2 главе пажње (2*2=4):\n']
        elif src.startswith('### 3.6.2 Implementing multi-head attention with weight splits'):
            nb['cells'][i]['source'] = ['### 3.6.2 Имплементација multi-head attention са поделом тежина\n']
        elif src.startswith('- While the above is an intuitive'):
            nb['cells'][i]['source'] = ['- Иако је горња имплементација интуитивна, можемо написати самосталну `MultiHeadAttention` класу:\n', '- Уместо спајања појединачних глава, креирамо једну W_query, W_key и W_value матрицу и делимо их за сваку главу:\n']
        elif src.startswith('- Note that the above is essentially a rewritten version'):
            nb['cells'][i]['source'] = ['- Горња имплементација је у суштини прерађена верзија `MultiHeadAttentionWrapper` која је ефикаснија:\n', '- Излази се мало разликују јер су почетне тежине различите, али обе су потпуно функционалне:\n']
        elif src.startswith('- Note that if you are interested in a compact'):
            nb['cells'][i]['source'] = ['- Ако сте заинтересовани за компактну имплементацију, можете користити [`torch.nn.MultiheadAttention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) класу у PyTorch-у:\n']
        elif src.startswith('- Since the above implementation may look a bit complex'):
            nb['cells'][i]['source'] = ['- Пошто горња имплементација може изгледати сложено, погледајмо шта се дешава при извршавању `attn_scores = queries @ keys.transpose(2, 3)`:\n']
        elif src.startswith('- In this case, the matrix multiplication'):
            nb['cells'][i]['source'] = ['- У овом случају, PyTorch ће руковати 4-димензионалним улазним тензором тако да се матрично множење изврши између последње 2 димензије:\n']
        elif src.startswith('- For instance, the following becomes'):
            nb['cells'][i]['source'] = ['- На пример, следеће постаје компактнији начин да се израчуна матрично множење за сваку главу посебно:\n']
            
        # --- Summary ---
        elif src.startswith('# Summary and takeaways'):
            nb['cells'][i]['source'] = ['# Резиме и закључци\n']
        elif src.startswith('- See the [./multihead-attention.ipynb]'):
            nb['cells'][i]['source'] = [
                '- Погледајте бележницу са кодом [./multihead-attention.ipynb](./multihead-attention.ipynb), која је сажета верзија учитавача података (поглавље 2) плус класа за multi-head attention коју смо имплементирали у овом поглављу.\n',
                '- Решења вежби можете наћи у [./exercise-solutions.ipynb](./exercise-solutions.ipynb).\n'
            ]

        # Remaining specific cells
        elif src.startswith('Packages that are being used'):
            pass  # already handled above

with open('ch03.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Превод ch03.ipynb је завршен!')
