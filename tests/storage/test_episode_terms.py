import pytest

from career_agent.storage.episode_terms import MAX_SEARCH_TERMS, search_terms, title_entities


def test_known_chinese_entity_stays_whole() -> None:
    assert "美团" in search_terms("我投过美团哪个岗位", ("美团",))


def test_english_name_and_compensation_stay_whole() -> None:
    assert search_terms("Pinnacle Robotics", ("Pinnacle Robotics",)) == (
        "pinnacle robotics",
    )
    assert search_terms("Senior Software Engineer", ()) == (
        "senior", "software", "engineer"
    )
    terms = search_terms("Pinnacle Robotics 35k×15 Temu PPO", ("Pinnacle Robotics",))
    assert "pinnacle robotics" in terms
    assert "35k×15" in terms
    assert "temu" in terms
    assert "ppo" in terms


def test_terms_are_bounded() -> None:
    assert len(search_terms("求职" * 100, ())) <= MAX_SEARCH_TERMS


@pytest.mark.parametrize("query", ["metadata", "metaverse", "metadata_store", "meta2", "metadata我投过"])
def test_english_entity_does_not_split_a_larger_word(query) -> None:
    assert search_terms(query, ("Meta",)) == search_terms(query, ())


@pytest.mark.parametrize("query", ["Meta", "(META)", "我投过Meta的岗位"])
def test_english_entity_at_a_word_boundary_is_preserved(query) -> None:
    assert "meta" in search_terms(query, ("Meta",))


def test_multiword_english_entity_cannot_match_a_longer_last_word() -> None:
    assert search_terms("Pinnacle RoboticsLab", ("Pinnacle Robotics",)) == (
        "pinnacle", "roboticslab",
    )


def test_empty_entity_dictionary_falls_back_to_fragments() -> None:
    terms = search_terms("我投过美团哪个岗位", ())
    assert "美团" in terms
    assert "投过美" in terms


def test_title_entities_drop_fixed_suffixes() -> None:
    entities = title_entities((
        "美团 · 配送算法工程师 · 第 2 轮面试",
        "腾讯 · 公司调研",
        "模拟面试：后台开发",
        "简历定制已完成",
    ))
    assert set(entities) == {"美团", "配送算法工程师", "腾讯", "后台开发"}
