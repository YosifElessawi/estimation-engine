"""Step 5's chart page: one page of plain charts, each with its mark to beat."""

from pathlib import Path

from estimation.onenet.charts import CHARTS, Charts, layout, values_for


def test_the_page_holds_every_chart_with_all_its_lines():
    page = layout()
    for section, title, lines in CHARTS:
        assert page[section][title] == ["Multiline", [tag for tag, _ in lines]]
    assert len({title for _, title, _ in CHARTS}) == len(CHARTS)


def test_a_loop_writes_its_numbers_and_the_marks_beside_them():
    assert values_for({"king_vs_copy": 0.3}) == {
        "strength/king_vs_copy": 0.3,
        "marks/even_king": 0.25,
    }
    assert values_for({"seed": 3.0}) == {}  # a number no chart shows draws nothing
    assert values_for({"loop": 40.0, "loops_total": 2000.0}) == {
        "progress/loops": 40.0,
        "marks/loops_total": 2000.0,
    }


def test_the_charts_are_written_as_tensorboard_files(tmp_path: Path):
    charts = Charts(tmp_path / "charts")
    charts.write(1, {"king_vs_copy": 0.26, "minutes_left": 120.0})
    charts.close()
    files = list((tmp_path / "charts").iterdir())
    assert len(files) == 1
    assert files[0].name.startswith("events.out.tfevents")
    assert files[0].stat().st_size > 0
