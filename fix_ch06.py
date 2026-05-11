import json

with open('ch06.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Map first 50 chars of untranslated cells to Serbian translations
# Each entry: (prefix, [list_of_lines])
trans = []

trans.append(('- Classification finetuning, the topic', [
    '- Класификационо фино подешавање, тема овог поглавља, је поступак који вам је можда познат ако имате искуства у машинском учењу',
    '- У класификационом фином подешавању, имамо одређени број ознака класа (нпр. "spam" и "not spam") које модел може да излази',
    '- Модел фино подешен за класификацију може предвидети само класе које је видео током тренирања',
    '- Можемо мислити о класификационом моделу као веома специјализованом моделу',
]))

trans.append(('- When we check the class distribution', [
    '- Када проверимо расподелу класа, видимо да подаци садрже "ham" (тј. "not spam") много чешће него "spam"',
]))

trans.append(('- For simplicity, and because we prefer', [
    '- Ради једноставности, подузоркујемо (undersample) скуп тако да садржи 747 примерака из сваке класе',
]))

trans.append(('- Next, we change the string class', [
    '- Затим мењамо текстуалне ознаке класа "ham" и "spam" у целобројне ознаке 0 и 1',
]))

trans.append(("- Let's now define a function", [
    '- Дефинишимо функцију која насумично дели скуп података на тренинг, валидациони и тест подскуп',
]))

trans.append(('- Note that the text messages', [
    '- Текстуалне поруке имају различите дужине; ако желимо да комбинујемо више примера у пакету, морамо',
    '1. скратити све поруке на дужину најкраће поруке у скупу или пакету',
    '2. допунити све поруке до дужине најдуже поруке у скупу података',
    '- Бирамо опцију 2 и допуњујемо све поруке до најдуже поруке у скупу',
    '- За то користимо [ENDOF] као токен за допуну',
]))

trans.append(("- Lastly, let's print", [
    '- На крају, одштампајмо укупан број пакета у сваком скупу',
]))

trans.append(("- To ensure that the model was loaded", [
    '- Да проверимо да ли је модел исправно учитан, проверимо да генерише кохерентан текст',
]))

trans.append(("- Before we finetune the model as a", [
    '- Пре него што фино подесимо модел као класификатор, проверимо да ли можда већ може класификовати спам поруке путем prompting-а',
]))

trans.append(('- In this section, we are modifying the pretrained', [
    '- У овом одељку модификујемо предтренирани LLM да га припремимо за класификационо фино подешавање',
    '- Прво погледајмо архитектуру модела',
]))

trans.append(("- Technically, it's sufficient", [
    '- Технички, довољно је тренирати само излазни слој',
    '- Међутим, експерименти показују да фино подешавање додатних слојева може значајно побољшати перформансе',
    '- Зато чинимо и последњи transformer блок и завршни LayerNorm модул тренирајућим',
]))

trans.append(("- We can still use this model", [
    '- Модел и даље можемо користити слично као раније',
    '- На пример, проследимо му неки текст као улаз',
]))

trans.append(("- What's different compared", [
    '- Разлика у односу на претходна поглавља је што сада има 2 излазне димензије уместо 50.257',
]))

trans.append(('- As discussed in previous chapters', [
    '- Као што је дискутовано у претходним поглављима, за сваки улазни токен постоји један излазни вектор',
    '- Пошто смо проследили моделу текст са 4 улазна токена, излаз се састоји од 4 2-димензионална вектора',
]))

trans.append(('- In chapter 3, we discussed the', [
    '- У поглављу 3 смо дискутовали о механизму пажње који повезује сваки улазни токен са сваким другим',
    '- Увели смо и каузалну маску пажње која се користи у GPT моделима',
    '- На основу овога, 4. (последњи) токен садржи највише информација',
    '- Зато смо посебно заинтересовани за овај последњи токен',
]))

trans.append(("- Before explaining the loss", [
    '- Пре објашњења израчунавања loss-а, погледајмо како се излази модела претварају у ознаке класа',
]))

trans.append(('- We can apply this concept', [
    '- Овај концепт можемо применити за израчунавање класификационе тачности',
]))

trans.append(("- Let's apply the function", [
    '- Применимо функцију за израчунавање тачности на различитим скуповима података',
]))

trans.append(("- As we can see, the prediction", [
    '- Као што видимо, тачности предвиђања нису добре јер још увек нисмо фино подесили модел',
]))

trans.append(('- Before we can start finetuning', [
    '- Пре него што почнемо фино подешавање, прво морамо дефинисати функцију губитка (loss function) коју желимо да оптимизујемо',
    '- Циљ је максимизовати тачност класификације спама; међутим, тачност није диференцијабилна функција',
    '- Стога минимизујемо cross-entropy loss као proxy за максимизацију тачности',
]))

trans.append(('- In this section, we define and use', [
    '- У овом одељку дефинишемо и користимо тренинг функцију да побољшамо тачност класификације',
    '- Функција train_classifier_simple је практично иста као train_model_simple из поглавља 5',
    '- Разлике су:',
    '    1. пратимо број виђених тренинг примера (examples_seen) уместо броја токена',
    '    2. израчунавамо тачност након сваке епохе уместо штампања узорка текста',
]))

trans.append(('- Above, based on the downward slope', [
    '- На основу силазне путање изнад, видимо да модел добро учи',
    '- Чињеница да су тренинг и валидациони loss веома близу указује да модел нема тенденцију претрениравања',
]))

trans.append(('- We can see that the training and validation', [
    '- Видимо да су тренинг и валидационе перформансе практично идентичне',
    '- Међутим, на основу нешто ниже тест тачности, видимо да се модел у малом степену претренирава',
    '- Ово је нормално и може се потенцијално смањити повећањем dropout стопе или weight_decay-а',
]))

trans.append(("- Finally, let's use the finetuned", [
    '- Коначно, хајде да употребимо фино подешени GPT модел у акцији',
    '- Функција classify_review имплементира кораке предобраде података сличне SpamDataset-у',
    '- Затим функција враћа предвиђену целобројну ознаку класе и одговарајуће име класе',
]))

trans.append(("- Let's try it out on a few examples", [
    '- Пробајмо на неколико примера испод',
]))

trans.append(("- Finally, let's save the model", [
    '- Коначно, сачувајмо модел за каснију употребу без поновног тренирања',
]))

# Also translate the chapter title
trans.append(('# Chapter 6: Finetuning for Text Classification', [
    '# Поглавље 6: Фино подешавање за класификацију текста',
]))

# Apply translations
matched_count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'markdown':
        continue
    src = ''.join(cell['source']).strip()
    if not src or src.startswith('<') or src.startswith('>') or src.startswith('---'):
        continue
    has_cyrillic = any(ord(c) > 0x0400 and ord(c) < 0x0500 for c in src)
    if has_cyrillic or len(src) < 15:
        continue
    
    for prefix, lines in trans:
        if src.startswith(prefix):
            nb['cells'][i]['source'] = [l + '\n' for l in lines]
            matched_count += 1
            break

with open('ch06.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Translated {matched_count} cells!')
