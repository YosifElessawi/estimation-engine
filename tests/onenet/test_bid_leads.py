"""Step 6's lead models: the leads they learn from, what they read from a view, suit shuffles,
folds, and the bids their leads pick."""

import numpy as np
import pytest

from estimation.onenet import bid_leads as D
from estimation.onenet import lessons as L
from estimation.onenet import suits as S

QUICK = D.Reading(size="32x2", epochs=150, batch=64, lr=3e-3, shuffle=False)
LOWER, HIGHER = 54, 55  # two estimates: the player's own and the one the teacher tried


def blank(count: int, slots: int = 3) -> L.Lessons:
    empty = L.empty(slots)
    lessons = {k: np.zeros((count, *empty[k].shape[1:]), L.DTYPES[k]) for k in L.DTYPES}
    lessons["options"][:] = -1
    return lessons


def estimates(views: np.ndarray, lead: np.ndarray, bola: int = 0) -> L.Lessons:
    """Lessons at estimate ``views`` under suit 0 as trump, where the teacher tried one other
    estimate and measured ``lead`` for it in a short exam."""
    lessons = blank(len(views), 2)
    lessons["game"][:] = bola
    lessons["x"][:] = views
    lessons["x"][:, S.TRUMP] = 1.0
    lessons["x"][:, S.LEGAL + LOWER] = lessons["x"][:, S.LEGAL + HIGHER] = 1.0
    lessons["phase"][:] = 2
    lessons["options"][:] = [LOWER, HIGHER]
    lessons["finalist"][:] = 1
    lessons["exam_deals"][:] = 4
    lessons["exam"][:] = lead
    lessons["exam_error"][:] = 1.0
    return lessons


def hands(count: int, seed: int) -> np.ndarray:
    """Views that show only a random hand of trumps (suit 0)."""
    views = np.zeros((count, L.VIEW), np.float16)
    views[:, :13] = np.random.default_rng(seed).random((count, 13)) < 0.5
    return views


# The lessons as leads to read.


def test_the_tried_slots_are_the_alternatives():
    lessons = blank(2)
    lessons["options"][:] = [[3, 7, 12], [3, 7, -1]]
    assert D.tried(lessons).tolist() == [[False, True, True], [False, True, False]]


def test_a_lead_counts_by_its_precision_and_the_finalists_is_its_exam():
    lessons = blank(2)
    lessons["options"][:] = [[3, 7, 12], [3, 7, -1]]
    lessons["margin"][:] = [[0.0, 1.0, -8.0], [0.0, -2.0, 0.0]]
    lessons["error"][:] = [[0.0, 2.0, 4.0], [0.0, 2.0, 0.0]]
    lessons["finalist"][:] = 1
    lessons["exam"][:] = [3.0, 9.0]
    lessons["exam_error"][:] = 1.0
    lessons["exam_deals"][:] = [4, 0]  # the second lesson's finalist never sat an exam
    aims = D.targets(lessons, floor=0.5)
    assert aims.options.tolist() == [[3, 7, 12], [3, 7, 0]]
    # The first finalist counts by its exam, the dropped option and the second finalist by the
    # deals they played; the own move and the unused slot do not count.
    assert aims.lead.tolist() == [[0.0, 3.0, -8.0], [0.0, -2.0, 0.0]]
    assert aims.weight[0].tolist() == pytest.approx([0.0, 1 / 1.25, 1 / 16.25])
    assert aims.weight[1].tolist() == pytest.approx([0.0, 1 / 4.25, 0.0])


# The model.


def test_a_lead_model_reads_a_lead_the_view_shows():
    views = hands(600, 1)
    lead = np.where(views[:, 5] > 0.5, 6.0, -6.0)  # the trump that decides it
    net = D.fit(estimates(views[:500], lead[:500]), QUICK, seed=2)
    fresh = estimates(views[500:], lead[500:])
    read = D.predict(net, fresh)
    assert read.shape == (100, 2)
    assert (read[:, 0] == 0).all()  # the own move leads itself by nothing
    assert (np.sign(read[:, 1]) == np.sign(lead[500:])).mean() > 0.95
    assert np.abs(read[:, 1] - lead[500:]).mean() < 2.0
    again = D.predict(D.fit(estimates(views[:500], lead[:500]), QUICK, seed=2), fresh)
    assert np.array_equal(read, again)  # the same seed fits the same model


def test_a_lead_model_gives_the_mean_lead_of_decisions_that_look_alike():
    # One view, taught 300 times: the option leads by 9 a third of the time and trails by 3
    # the rest. Each time it trails, yet on average it leads by 1.
    views = np.zeros((300, L.VIEW), np.float16)
    lead = np.where(np.arange(300) % 3 == 0, 9.0, -3.0)
    net = D.fit(estimates(views, lead), QUICK, seed=3)
    assert D.predict(net, estimates(views[:1], lead[:1]))[0, 1] == pytest.approx(1.0, abs=0.5)


def test_suit_shuffles_teach_an_estimate_under_every_renaming_of_the_other_suits():
    ace_of = [S.HAND + 13 * suit + 12 for suit in range(4)]
    views = hands(400, 4)
    held = np.arange(400) % 2 == 0
    views[held, ace_of[1]] = 1.0  # every lesson's ace is in suit 1
    taught = estimates(views, np.where(held, 8.0, -8.0))
    fresh = hands(50, 5)
    fresh[:, ace_of[2]] = 1.0  # the same hands with the ace in suit 2
    probe = estimates(fresh, np.zeros(50))
    plain = D.predict(D.fit(taught, QUICK, seed=6), probe)[:, 1]
    shuffled_reading = D.Reading(size="32x2", epochs=150, batch=64, lr=3e-3, shuffle=True)
    shuffled = D.predict(D.fit(taught, shuffled_reading, seed=6), probe)[:, 1]
    assert (shuffled > 4.0).mean() > 0.9  # an ace is an ace in any suit but the trump
    assert (plain < 0.0).mean() > 0.9  # without shuffles suit 2's ace was never seen


# Folds.


def test_a_fold_holds_whole_bolas():
    games = np.repeat(np.arange(10), 3)
    fold = D.fold_of(games, 5, seed=1)
    assert sorted(np.bincount(fold).tolist()) == [6] * 5  # two Bolas of three lessons a fold
    for game in range(10):
        assert len(set(fold[games == game].tolist())) == 1
    assert D.fold_of(games, 5, seed=1).tolist() == fold.tolist()
    assert D.fold_of(games, 5, seed=2).tolist() != fold.tolist()
    with pytest.raises(ValueError, match="folds"):
        D.fold_of(games, 11, seed=1)
    with pytest.raises(ValueError, match="folds"):
        D.fold_of(games, 1, seed=1)


def test_every_training_lesson_is_read_by_a_model_that_never_saw_its_bola():
    # Every view is the same, and the lead is the Bola's own: +10 in Bola 0, -10 in Bola 1.
    # Nothing in a view says which Bola it is, so a model can only repeat the Bola it was fit
    # on: each Bola's lessons come out with the other Bola's lead.
    views = np.zeros((40, L.VIEW), np.float16)
    first, second = estimates(views, np.full(40, 10.0), 0), estimates(views, np.full(40, -10.0), 1)
    train = L.join([first, second])
    held = estimates(views[:5], np.zeros(5), 2)
    said: list[str] = []
    read, held_read = D.cross_fit(train, held, QUICK, folds=2, seed=1, say=said.append)
    assert read[:40, 1] == pytest.approx(-10.0, abs=1.0)
    assert read[40:, 1] == pytest.approx(10.0, abs=1.0)
    assert held_read[:, 1] == pytest.approx(0.0, abs=1.0)  # both models' mean
    assert (read[:, 0] == 0).all()
    assert sum("lead model" in line for line in said) == 2


# The bids the leads pick.


def test_the_leads_pick_the_option_read_furthest_ahead_past_the_margin():
    lessons = blank(4)
    lessons["options"][:] = [[3, 7, 12], [3, 7, 12], [3, 7, 12], [3, 7, -1]]
    leads = np.array(
        [[0.0, 0.4, 0.2], [0.0, 1.0, 3.0], [0.0, 2.0, -1.0], [0.0, -1.0, 9.0]], np.float32
    )
    # Nothing clears half a point; then the furthest ahead; an unused slot is never picked.
    assert D.picks(leads, lessons, margin=0.5).tolist() == [3, 12, 7, 3]
    assert D.picks(leads, lessons, margin=2.5).tolist() == [3, 12, 3, 3]


def test_a_reading_is_checked():
    with pytest.raises(ValueError, match="units x layers"):
        D.Reading(size="big")
    with pytest.raises(ValueError, match="epochs"):
        D.Reading(epochs=0)
    with pytest.raises(ValueError, match="lr"):
        D.Reading(lr=0.0)
    with pytest.raises(ValueError, match="lessons to read"):
        D.fit(blank(0), QUICK, seed=0)
