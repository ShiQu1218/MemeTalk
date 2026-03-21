from memetalk.app.ui_state import resolve_index_folder_input


def test_index_folder_input_uses_saved_folder_on_first_load() -> None:
    assert resolve_index_folder_input("D:/SavedMemes", None, None) == "D:/SavedMemes"


def test_index_folder_input_refreshes_when_user_has_not_edited_old_default() -> None:
    assert (
        resolve_index_folder_input(
            saved_folder="D:/NewDefault",
            current_input="D:/OldDefault",
            last_saved_folder="D:/OldDefault",
        )
        == "D:/NewDefault"
    )


def test_index_folder_input_preserves_user_draft_when_saved_default_changes() -> None:
    assert (
        resolve_index_folder_input(
            saved_folder="D:/NewDefault",
            current_input="D:/ScratchRun",
            last_saved_folder="D:/OldDefault",
        )
        == "D:/ScratchRun"
    )
