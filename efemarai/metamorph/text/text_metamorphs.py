from collections.abc import Callable

from efemarai.metamorph.adaptors import apply_nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from nlpaug.util import Action

from efemarai.metamorph.base_metamorphs import def_operator, param, siso, Category


@def_operator(category=Category.Text)
@param("word_synonym_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def WordSynonym(word_synonym_p) -> Callable:
    if word_synonym_p == 0:
        return lambda text: [text]

    aug = naw.SynonymAug(aug_p=word_synonym_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_antonym_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def WordAntonym(word_antonym_p) -> Callable:
    if word_antonym_p == 0:
        return lambda text: [text]

    aug = naw.AntonymAug(aug_p=word_antonym_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_contextual_embs_p", float, range=(0, 1), value=0.3, fixed=True)
@param(
    "action",
    str,
    choices=("substitute", "insert"),
    fixed=True,
    value="substitute",
)
@siso()
@apply_nlpaug()
def WordContextualEmbs(word_contextual_embs_p, action) -> Callable:
    if word_contextual_embs_p == 0:
        return lambda text: [text]

    aug = naw.ContextualWordEmbsAug(
        aug_p=word_contextual_embs_p, aug_min=None, aug_max=None, action=action
    )
    return aug.augment


@def_operator(category=Category.Text)
@param("word_random_p", float, range=(0, 1), value=0.3, fixed=True)
@param(
    "action",
    str,
    choices=(Action.DELETE, Action.SUBSTITUTE, Action.SWAP, Action.CROP),
    fixed=True,
    value=Action.DELETE,
)
@siso()
@apply_nlpaug()
def WordRandom(word_random_p, action) -> Callable:
    if word_random_p == 0:
        return lambda text: [text]

    aug = naw.RandomWordAug(
        aug_p=word_random_p, aug_min=None, aug_max=None, action=action
    )
    return aug.augment


@def_operator(category=Category.Text)
@param("word_spelling_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def WordSpelling(word_spelling_p) -> Callable:
    if word_spelling_p == 0:
        return lambda text: [text]

    aug = naw.SpellingAug(aug_p=word_spelling_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_split_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def WordSplit(word_split_p) -> Callable:
    if word_split_p == 0:
        return lambda text: [text]

    aug = naw.SplitAug(aug_p=word_split_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_tfidf_p", float, range=(0, 1), value=0.3, fixed=True)
@param(
    "action",
    str,
    choices=(Action.SUBSTITUTE, Action.INSERT),
    fixed=True,
    value=Action.SUBSTITUTE,
)
@siso()
@apply_nlpaug()
def WordTfIdf(word_tfidf_p, action) -> Callable:
    if word_tfidf_p == 0:
        return lambda text: [text]

    aug = naw.TfIdfAug(aug_p=word_tfidf_p, aug_max=None, action=action)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_embs_p", float, range=(0, 1), value=0.3, fixed=True)
@param(
    "action",
    str,
    choices=(Action.SUBSTITUTE, Action.INSERT),
    fixed=True,
    value=Action.SUBSTITUTE,
)
@siso()
@apply_nlpaug()
def WordEmbeddings(word_embs_p) -> Callable:
    if word_embs_p == 0:
        return lambda text: [text]

    aug = naw.WordEmbsAug(aug_p=word_embs_p, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_back_translation", bool, choices=(False, True), fixed=True)
@siso()
@apply_nlpaug()
def WordBackTranslation(word_back_translation: bool) -> Callable:
    if not word_back_translation:
        return lambda text: [text]

    aug = naw.BackTranslationAug()
    return aug.augment


@def_operator(category=Category.Text)
@param("word_reserved_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def WordReserved(word_reserved_p) -> Callable:
    if word_reserved_p == 0:
        return lambda text: [text]

    aug = naw.ReservedAug(aug_p=word_reserved_p, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("char_keyboard_char_p", float, range=(0, 1), value=0.3, fixed=True)
@param("char_keyboard_word_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def CharKeyboard(char_keyboard_char_p, char_keyboard_word_p) -> Callable:
    if char_keyboard_char_p == 0 or char_keyboard_word_p == 0:
        return lambda text: [text]

    aug = nac.KeyboardAug(
        aug_char_p=char_keyboard_char_p,
        aug_word_p=char_keyboard_word_p,
        aug_char_max=None,
        aug_word_max=None,
    )
    return aug.augment


@def_operator(category=Category.Text)
@param("char_random_char_p", float, range=(0, 1), value=0.3, fixed=True)
@param("char_random_word_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def CharRandom(char_random_char_p, char_random_word_p) -> Callable:
    if char_random_char_p == 0 or char_random_word_p == 0:
        return lambda text: [text]

    aug = nac.RandomCharAug(
        aug_char_p=char_random_char_p,
        aug_word_p=char_random_word_p,
        aug_char_max=None,
        aug_word_max=None,
    )
    return aug.augment


@def_operator(category=Category.Text)
@param("char_ocr_char_p", float, range=(0, 1), value=0.3, fixed=True)
@param("char_ocr_word_p", float, range=(0, 1), value=0.3, fixed=True)
@siso()
@apply_nlpaug()
def CharOcr(char_ocr_char_p, char_ocr_word_p) -> Callable:
    if char_ocr_char_p == 0 or char_ocr_word_p == 0:
        return lambda text: [text]

    aug = nac.OcrAug(
        aug_char_p=char_ocr_char_p,
        aug_word_p=char_ocr_word_p,
        aug_char_max=None,
        aug_word_max=None,
    )
    return aug.augment


@def_operator(category=Category.Text)
@param("sentence_abst_summ", bool, choices=(False, True), fixed=True)
@siso()
@apply_nlpaug()
def SentenceAbstSumm(sentence_abst_summ: bool) -> Callable:
    if not sentence_abst_summ:
        return lambda text: [text]

    aug = nas.AbstSummAug()
    return aug.augment


@def_operator(category=Category.Text)
@param("sentence_lambada", bool, choices=(False, True), fixed=True)
@siso()
@apply_nlpaug()
def SentenceLambada(sentence_lambada: bool) -> Callable:
    if not sentence_lambada:
        return lambda text: [text]

    aug = nas.LambadaAug()
    return aug.augment


@def_operator(category=Category.Text)
@param("sentence_contextual_word_embs", bool, choices=(False, True), fixed=True)
@siso()
@apply_nlpaug()
def SentenceContextualWordEmbs(sentence_contextual_word_embs: bool) -> Callable:
    if not sentence_contextual_word_embs:
        return lambda text: [text]

    aug = nas.ContextualWordEmbsForSentenceAug()
    return aug.augment
