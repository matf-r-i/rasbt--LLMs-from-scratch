import json

with open('ch06.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

translations = {
    # --- 6.1 ---
    "## 6.1 Different categories of finetuning": "## 6.1 Различите категорије фино подешавања",
    "- No code in this section": "- Нема кода у овом одељку",
    "- The most common ways to finetune language models are instruction-finetuning and classification finetuning": "- Најчешћи начини фино подешавања језичких модела су инструкционо фино подешавање (instruction-finetuning) и класификационо фино подешавање (classification finetuning)",
    "- Instruction-finetuning, depicted below, is the topic of the next chapter": "- Инструкционо фино подешавање, приказано испод, тема је следећег поглавља",
    "- Classification finetuning, the topic of this chapter, is a procedure you may already be familiar with if you have a background in machine learning -- it's similar to training a convolutional network to classify handwritten digits, for example": "- Класификационо фино подешавање, тема овог поглавља, је поступак који вам је можда познат ако имате искуства у машинском учењу",
    '- In classification finetuning, we have a specific number of class labels (for example, "spam" and "not spam") that the model can output': '- У класификационом фином подешавању, имамо одређени број ознака класа (нпр. "spam" и "not spam") које модел може да излази',
    '- A classification finetuned model can only predict classes it has seen during training (for example, "spam" or "not spam"), whereas an instruction-finetuned model can usually perform many tasks': '- Модел фино подешен за класификацију може предвидети само класе које је видео током тренирања, док инструкционо фино подешен модел обично може обављати много задатака',
    "- We can think of a classification-finetuned model as a very specialized model; in practice, it is much easier to create a specialized model than a generalist model that performs well on many different tasks": "- Можемо мислити о класификационом моделу као веома специјализованом моделу; у пракси је много лакше направити специјализовани него генералистички модел",
    # --- 6.2 ---
    "## 6.2 Preparing the dataset": "## 6.2 Припрема скупа података",
    "- This section prepares the dataset we use for classification finetuning": "- Овај одељак припрема скуп података који користимо за класификационо фино подешавање",
    "- We use a dataset consisting of spam and non-spam text messages to finetune the LLM to classify them": "- Користимо скуп података који се састоји од спам и не-спам текстуалних порука",
    "- First, we download and unzip the dataset": "- Прво преузимамо и отпакујемо скуп података",
    "- The dataset is saved as a tab-separated text file, which we can load into a pandas DataFrame": "- Скуп података је сачуван као табулатором раздвојена текст датотека, коју можемо учитати у pandas DataFrame",
    '- When we check the class distribution, we see that the data contains "ham" (i.e., "not spam") much more frequently than "spam"': '- Када проверимо расподелу класа, видимо да подаци садрже "ham" (тј. "not spam") много чешће него "spam"',
    "- For simplicity, and because we prefer a small dataset for educational purposes anyway (it will make it possible to finetune the LLM faster), we subsample (undersample) the dataset so that it contains 747 instances from each class": "- Ради једноставности, подузоркујемо скуп тако да садржи 747 примерака из сваке класе",
    '- Next, we change the string class labels "ham" and "spam" into integer class labels 0 and 1': '- Затим мењамо текстуалне ознаке класа "ham" и "spam" у целобројне ознаке 0 и 1',
    "- Let's now define a function that randomly divides the dataset into training, validation, and test subsets": "- Дефинишимо функцију која насумично дели скуп података на тренинг, валидациони и тест подскуп",
    # --- 6.3 ---
    "## 6.3 Creating data loaders": "## 6.3 Креирање data loader-а",
    "- Note that the text messages have different lengths; if we want to combine multiple training examples in a batch, we have to either": "- Текстуалне поруке имају различите дужине; ако желимо да комбинујемо више примера у пакету, морамо",
    "1. truncate all messages to the length of the shortest message in the dataset or batch": "1. скратити све поруке на дужину најкраће поруке у скупу или пакету",
    "2. pad all messages to the length of the longest message in the dataset or batch": "2. допунити све поруке до дужине најдуже поруке у скупу података",
    "- We choose option 2 and pad all messages to the longest message in the dataset": "- Бирамо опцију 2 и допуњујемо све поруке до најдуже поруке у скупу",
    "- For that, we use `<|endoftext|>` as a padding token, as discussed in chapter 2": "- За то користимо `<|endoftext|>` као токен за допуну",
    "- The `SpamDataset` class below identifies the longest sequence in the training dataset and adds the padding token to the others to match that sequence length": "- Класа `SpamDataset` испод идентификује најдужу секвенцу у тренинг скупу и додаје токен за допуну осталима",
    "- We also pad the validation and test set to the longest training sequence": "- Такође допуњујемо валидациони и тест скуп до најдуже тренинг секвенце",
    "- Note that validation and test set samples that are longer than the longest training example are being truncated via `encoded_text[:self.max_length]` in the `SpamDataset` code": "- Имајте на уму да се валидациони и тест примерци дужи од најдужег тренинг примера скраћују",
    "- This behavior is entirely optional, and it would also work well if we set `max_length=None` in both the validation and test set cases": "- Ово понашање је потпуно опционо",
    "- Next, we use the dataset to instantiate the data loaders, which is similar to creating the data loaders in previous chapters": "- Затим користимо скуп података за инстанцирање data loader-а",
    "- As a verification step, we iterate through the data loaders and ensure that the batches contain 8 training examples each, where each training example consists of 120 tokens": "- Корак провере: пролазимо кроз data loader-е и проверавамо да пакети садрже по 8 тренинг примера",
    "- Lastly, let's print the total number of batches in each dataset": "- На крају, одштампајмо укупан број пакета у сваком скупу",
    # --- 6.4 ---
    "## 6.4 Initializing a model with pretrained weights": "## 6.4 Иницијализација модела са предтренираним тежинама",
    "- In this section, we initialize the pretrained model we worked with in the previous chapter": "- У овом одељку иницијализујемо предтренирани модел са којим смо радили у претходном поглављу",
    "- To ensure that the model was loaded correctly, let's double-check that it generates coherent text": "- Да проверимо да ли је модел исправно учитан, проверимо да генерише кохерентан текст",
    "- Before we finetune the model as a classifier, let's see if the model can perhaps already classify spam messages via prompting": "- Пре него што фино подесимо модел као класификатор, проверимо да ли можда већ може класификовати спам поруке путем prompting-а",
    "- As we can see, the model is not very good at following instructions": "- Као што видимо, модел није баш добар у праћењу инструкција",
    "- This is expected, since it has only been pretrained and not instruction-finetuned (instruction finetuning will be covered in the next chapter)": "- Ово је очекивано, јер је само предтрениран и није инструкционо фино подешен",
    # --- 6.5 ---
    "## 6.5 Adding a classification head": "## 6.5 Додавање класификационе главе (classification head)",
    "- In this section, we are modifying the pretrained LLM to make it ready for classification finetuning": "- У овом одељку модификујемо предтренирани LLM да га припремимо за класификационо фино подешавање",
    "- Let's take a look at the model architecture first": "- Прво погледајмо архитектуру модела",
    "- Above, we can see the architecture we implemented in chapter 4 neatly laid out": "- Горе видимо архитектуру коју смо имплементирали у поглављу 4",
    "- The goal is to replace and finetune the output layer": "- Циљ је заменити и фино подесити излазни слој",
    "- To achieve this, we first freeze the model, meaning that we make all layers non-trainable": "- Да бисмо то постигли, прво замрзавамо модел, тј. чинимо све слојеве нетренирајућим",
    '- Then, we replace the output layer (`model.out_head`), which originally maps the layer inputs to 50,257 dimensions (the size of the vocabulary)': '- Затим замењујемо излазни слој (`model.out_head`) који је оригинално мапирао улазе у 50.257 димензија',
    '- Since we finetune the model for binary classification (predicting 2 classes, "spam" and "not spam"), we can replace the output layer as shown below, which will be trainable by default': '- Пошто фино подешавамо модел за бинарну класификацију (2 класе), замењујемо излазни слој',
    '- Note that we use `BASE_CONFIG["emb_dim"]` (which is equal to 768 in the `"gpt2-small (124M)"` model) to keep the code below more general': '- Користимо `BASE_CONFIG["emb_dim"]` да код буде општији',
    "- Technically, it's sufficient to only train the output layer": "- Технички, довољно је тренирати само излазни слој",
    "- However, as I found in [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models), experiments show that finetuning additional layers can noticeably improve the performance": "- Међутим, експерименти показују да фино подешавање додатних слојева може значајно побољшати перформансе",
    "- So, we are also making the last transformer block and the final `LayerNorm` module connecting the last transformer block to the output layer trainable": "- Зато чинимо и последњи transformer блок и завршни `LayerNorm` модул тренирајућим",
    "- We can still use this model similar to before in previous chapters": "- Модел и даље можемо користити слично као раније",
    "- For example, let's feed it some text input": "- На пример, проследимо му неки текст као улаз",
    "- What's different compared to previous chapters is that it now has two output dimensions instead of 50,257": "- Разлика у односу на претходна поглавља је што сада има 2 излазне димензије уместо 50.257",
    "- As discussed in previous chapters, for each input token, there's one output vector": "- Као што је дискутовано у претходним поглављима, за сваки улазни токен постоји један излазни вектор",
    "- Since we fed the model a text sample with 4 input tokens, the output consists of 4 2-dimensional output vectors above": "- Пошто смо проследили моделу текст са 4 улазна токена, излаз се састоји од 4 2-димензионална вектора",
    "- In chapter 3, we discussed the attention mechanism, which connects each input token to each other input token": "- У поглављу 3 смо дискутовали о механизму пажње који повезује сваки улазни токен са сваким другим",
    "- In chapter 3, we then also introduced the causal attention mask that is used in GPT-like models; this causal mask lets a current token only attend to the current and previous token positions": "- Увели смо и каузалну маску пажње која се користи у GPT моделима",
    "- Based on this causal attention mechanism, the 4th (last) token contains the most information among all tokens because it's the only token that includes information about all other tokens": "- На основу овога, 4. (последњи) токен садржи највише информација",
    "- Hence, we are particularly interested in this last token, which we will finetune for the spam classification task": "- Зато смо посебно заинтересовани за овај последњи токен",
    # --- 6.6 ---
    "## 6.6 Calculating the classification loss and accuracy": "## 6.6 Израчунавање класификационог loss-а и тачности",
    "- Before explaining the loss calculation, let's have a brief look at how the model outputs are turned into class labels": "- Пре објашњења израчунавања loss-а, погледајмо како се излази модела претварају у ознаке класа",
    "- Similar to chapter 5, we convert the outputs (logits) into probability scores via the `softmax` function and then obtain the index position of the largest probability value via the `argmax` function": "- Слично поглављу 5, конвертујемо излазе у вероватноће путем softmax функције, а затим добијамо индекс највеће вероватноће путем argmax функције",
    "- Note that the softmax function is optional here, as explained in chapter 5, because the largest outputs correspond to the largest probability scores": "- Имајте на уму да је softmax функција овде опциона, јер највећи излази одговарају највећим вероватноћама",
    "- We can apply this concept to calculate the so-called classification accuracy, which computes the percentage of correct predictions in a given dataset": "- Овај концепт можемо применити за израчунавање класификационе тачности",
    "- Let's apply the function to calculate the classification accuracies for the different datasets": "- Применимо функцију за израчунавање тачности на различитим скуповима података",
    "- As we can see, the prediction accuracies are not very good, since we haven't finetuned the model, yet": "- Као што видимо, тачности предвиђања нису добре јер још увек нисмо фино подесили модел",
    "- Before we can start finetuning (/training), we first have to define the loss function we want to optimize during training": "- Пре него што почнемо фино подешавање, прво морамо дефинисати функцију губитка (loss function) коју желимо да оптимизујемо",
    "- The goal is to maximize the spam classification accuracy of the model; however, classification accuracy is not a differentiable function": "- Циљ је максимизовати тачност класификације спама; међутим, тачност није диференцијабилна функција",
    "- Hence, instead, we minimize the cross-entropy loss as a proxy for maximizing the classification accuracy (you can learn more about this topic in lecture 8 of my freely available [Introduction to Deep Learning](https://sebastianraschka.com/blog/2021/dl-course.html#l08-multinomial-logistic-regression--softmax-regression) class)": "- Стога минимизујемо cross-entropy loss као proxy за максимизацију тачности",
    "- The `calc_loss_batch` function is the same here as in chapter 5, except that we are only interested in optimizing the last token `model(input_batch)[:, -1, :]` instead of all tokens `model(input_batch)`": "- Функција `calc_loss_batch` је иста као у поглављу 5, осим што оптимизујемо само последњи токен уместо свих",
    "The `calc_loss_loader` is exactly the same as in chapter 5": "- `calc_loss_loader` је потпуно иста као у поглављу 5",
    "- Using the `calc_closs_loader`, we compute the initial training, validation, and test set losses before we start training": "- Користећи `calc_loss_loader`, израчунавамо почетне тренинг, валидационе и тест loss-ове пре почетка тренирања",
    "- In the next section, we train the model to improve the loss values and consequently the classification accuracy": "- У следећем одељку тренирамо модел да побољшамо вредности loss-а и последично тачност класификације",
    # --- 6.7 ---
    "## 6.7 Finetuning the model on supervised data": "## 6.7 Фино подешавање модела на надгледаним подацима",
    "- In this section, we define and use the training function to improve the classification accuracy of the model": "- У овом одељку дефинишемо и користимо тренинг функцију да побољшамо тачност класификације",
    "- The `train_classifier_simple` function below is practically the same as the `train_model_simple` function we used for pretraining the model in chapter 5": "- Функција `train_classifier_simple` је практично иста као `train_model_simple` из поглавља 5",
    "- The only two differences are that we now": "- Разлике су:",
    "1. track the number of training examples seen (`examples_seen`) instead of the number of tokens seen": "1. пратимо број виђених тренинг примера (`examples_seen`) уместо броја токена",
    "2. calculate the accuracy after each epoch instead of printing a sample text after each epoch": "2. израчунавамо тачност након сваке епохе уместо штампања узорка текста",
    "- The `evaluate_model` function used in the `train_classifier_simple` is the same as the one we used in chapter 5": "- Функција `evaluate_model` је иста као у поглављу 5",
    "- The training takes about 5 minutes on a M3 MacBook Air laptop computer and less than half a minute on a V100 or A100 GPU": "- Тренирање траје око 5 минута на M3 MacBook Air лаптопу и мање од пола минута на V100 или A100 GPU-у",
    "- Similar to chapter 5, we use matplotlib to plot the loss function for the training and validation set": "- Слично поглављу 5, користимо matplotlib за цртање функције loss-а за тренинг и валидациони скуп",
    "- Above, based on the downward slope, we see that the model learns well": "- На основу силазне путање изнад, видимо да модел добро учи",
    "- Furthermore, the fact that the training and validation loss are very close indicates that the model does not tend to overfit the training data": "- Чињеница да су тренинг и валидациони loss веома близу указује да модел нема тенденцију претрениравања",
    "- Similarly, we can plot the accuracy below": "- Слично, можемо приказати тачност испод",
    "- Based on the accuracy plot above, we can see that the model achieves a relatively high training and validation accuracy after epochs 4 and 5": "- На основу графика тачности, видимо да модел постиже релативно високу тренинг и валидациону тачност након епоха 4 и 5",
    "- However, we have to keep in mind that we specified `eval_iter=5` in the training function earlier, which means that we only estimated the training and validation set performances": "- Међутим, морамо имати на уму да смо поставили `eval_iter=5`",
    "- We can compute the training, validation, and test set performances over the complete dataset as follows below": "- Израчунајмо тачност на комплетним скуповима",
    "- We can see that the training and validation set performances are practically identical": "- Видимо да су тренинг и валидационе перформансе практично идентичне",
    "- However, based on the slightly lower test set performance, we can see that the model overfits the training data to a very small degree, as well as the validation data that has been used for tweaking some of the hyperparameters, such as the learning rate": "- Међутим, на основу нешто ниже тест тачности, видимо да се модел у малом степену претренирава",
    "- This is normal, however, and this gap could potentially be further reduced by increasing the model's dropout rate (`drop_rate`) or the `weight_decay` in the optimizer setting": "- Ово је нормално и може се потенцијално смањити повећањем dropout стопе или `weight_decay`-а",
    # --- 6.8 ---
    "## 6.8 Using the LLM as a spam classifier": "## 6.8 Коришћење LLM-а као класификатора спама",
    "- Finally, let's use the finetuned GPT model in action": "- Коначно, хајде да употребимо фино подешени GPT модел у акцији",
    "- The `classify_review` function below implements the data preprocessing steps similar to the `SpamDataset` we implemented earlier": "- Функција `classify_review` имплементира кораке предобраде података сличне `SpamDataset`-у",
    "- Then, the function returns the predicted integer class label from the model and returns the corresponding class name": "- Затим функција враћа предвиђену целобројну ознаку класе и одговарајуће име класе",
    "- Let's try it out on a few examples below": "- Пробајмо на неколико примера испод",
    "- Finally, let's save the model in case we want to reuse the model later without having to train it again": "- Коначно, сачувајмо модел за каснију употребу без поновног тренирања",
    "- Then, in a new session, we could load the model as follows": "- Затим, у новој сесији, можемо учитати модел на следећи начин",
    # --- Summary ---
    "## Summary and takeaways": "## Резиме и закључци",
    "- See the [./gpt_class_finetune.py](./gpt_class_finetune.py) script, a self-contained script for classification finetuning": "- Погледајте скрипту [./gpt_class_finetune.py](./gpt_class_finetune.py) за самостално класификационо фино подешавање",
    "- You can find the exercise solutions in [./exercise-solutions.ipynb](./exercise-solutions.ipynb)": "- Решења вежби можете наћи у [./exercise-solutions.ipynb](./exercise-solutions.ipynb)",
    "- In addition, interested readers can find an introduction to parameter-efficient training with low-rank adaptation (LoRA) in [appendix E](../../appendix-E)": "- Заинтересовани читаоци могу наћи увод у parameter-efficient тренирање са LoRA-ом у додатку E",
}

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source']).strip()
        
        if src in translations:
            new_text = translations[src]
            nb['cells'][i]['source'] = [new_text + '\n']
            continue
        
        # Handle multi-line cells
        lines = src.split('\n')
        if lines and len(lines) > 1:
            first_line = lines[0].strip()
            if first_line in translations:
                new_lines = [translations[first_line]]
                for line in lines[1:]:
                    stripped = line.strip()
                    if stripped in translations:
                        new_lines.append(translations[stripped])
                    elif stripped.startswith('- ') and stripped not in translations:
                        new_lines.append(stripped)
                    else:
                        new_lines.append(stripped)
                nb['cells'][i]['source'] = [l + ('\n' if not l.endswith('\n') else '') for l in new_lines]

with open('ch06.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Превод ch06.ipynb је завршен!')
