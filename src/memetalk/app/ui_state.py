from __future__ import annotations


def resolve_index_folder_input(
    saved_folder: str,
    current_input: str | None,
    last_saved_folder: str | None,
) -> str:
    """Keep the saved folder as the initial value without overwriting user drafts."""
    if current_input is None:
        return saved_folder
    if last_saved_folder is not None and current_input == last_saved_folder and saved_folder != last_saved_folder:
        return saved_folder
    return current_input
