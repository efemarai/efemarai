from collections.abc import Callable

from efemarai.metamorph.adaptors import apply_nlpaug
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
from nlpaug.util import Action

from efemarai.metamorph.base_metamorphs import def_operator, param, siso, Category


@def_operator(category=Category.Text)
@param("word_synonym_p", float, range=(0, 1))
@siso()
@apply_nlpaug()
def WordSynonym(word_synonym_p) -> Callable:
    if word_synonym_p == 0:
        return lambda text: [text]

    aug = naw.SynonymAug(aug_p=word_synonym_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_antonym_p", float, range=(0, 1))
@siso()
@apply_nlpaug()
def WordAntonym(word_antonym_p) -> Callable:
    if word_antonym_p == 0:
        return lambda text: [text]

    aug = naw.AntonymAug(aug_p=word_antonym_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_contextual_embs_p", float, range=(0, 1))
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
        aug_p=word_contextual_embs_p, aug_max=None, action=action
    )
    return aug.augment


# TODO: Fix edge cases with action = crop and word_random_p < 0.15
@def_operator(category=Category.Text)
@param("word_random_p", float, range=(0, 1))
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
@param("word_spelling_p", float, range=(0, 1))
@siso()
@apply_nlpaug()
def WordSpelling(word_spelling_p) -> Callable:
    if word_spelling_p == 0:
        return lambda text: [text]

    aug = naw.SpellingAug(aug_p=word_spelling_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_split_p", float, range=(0, 1))
@siso()
@apply_nlpaug()
def WordSplit(word_split_p) -> Callable:
    if word_split_p == 0:
        return lambda text: [text]

    aug = naw.SplitAug(aug_p=word_split_p, aug_min=None, aug_max=None)
    return aug.augment


@def_operator(category=Category.Text)
@param("word_back_translation", bool, choices=(False, True), value=True, fixed=True)
@siso()
@apply_nlpaug()
def WordBackTranslation(word_back_translation) -> Callable:
    if not word_back_translation:
        return lambda text: [text]

    aug = naw.BackTranslationAug()
    return aug.augment


@def_operator(category=Category.Text)
@param("char_keyboard_char_p", float, range=(0, 1))
@param("char_keyboard_word_p", float, range=(0, 1))
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
@param("char_random_char_p", float, range=(0, 1))
@param("char_random_word_p", float, range=(0, 1))
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
@param("char_ocr_char_p", float, range=(0, 1))
@param("char_ocr_word_p", float, range=(0, 1))
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
@param("sentence_abst_summ", bool, choices=(False, True), value=True, fixed=True)
@siso()
@apply_nlpaug()
def SentenceAbstSumm(sentence_abst_summ: bool) -> Callable:
    if not sentence_abst_summ:
        return lambda text: [text]

    aug = nas.AbstSummAug()
    return aug.augment


@def_operator(category=Category.Text)
@param("sentence_context_embs", bool, choices=(False, True), value=True, fixed=True)
@siso()
@apply_nlpaug()
def SentenceContextEmbs(sentence_context_embs: bool) -> Callable:
    if not sentence_context_embs:
        return lambda text: [text]

    aug = nas.ContextualWordEmbsForSentenceAug()
    return aug.augment