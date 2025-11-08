"""PDF Context Manager - Ensures PDFs are available as context for subsequent subtasks."""

import logging
from typing import Any, Dict, List, Optional

from ..models import Attachment
from ..state import InformationStateManager, Subtask


class PDFContextManager:
    """Manages PDF context passing between subtasks."""

    def __init__(
        self,
        state_manager: InformationStateManager,
        logger: logging.Logger,
    ):
        """
        Initialize PDF Context Manager.

        Args:
            state_manager: Information state manager.
            logger: Logger instance.
        """
        self.state_manager = state_manager
        self.logger = logger

    def get_pdfs_from_previous_subtasks(
        self, current_subtask_id: str
    ) -> List[Attachment]:
        """
        Get all PDFs downloaded in previous subtasks.

        Args:
            current_subtask_id: ID of the current subtask.

        Returns:
            List of PDF attachments from previous subtasks.
        """
        pdfs = []

        # Extract step number from current subtask ID (e.g., "step_3" -> 3)
        try:
            current_step_num = (
                int(current_subtask_id.split('_')[1])
                if '_' in current_subtask_id
                else None
            )
        except (ValueError, IndexError):
            current_step_num = None

        if current_step_num is not None:
            # Get all previous steps
            for step_num in range(1, current_step_num):
                prev_step_id = f'step_{step_num}'
                if prev_step_id in self.state_manager.subtasks:
                    prev_subtask = self.state_manager.subtasks[prev_step_id]
                    if prev_subtask.status == 'completed':
                        # Extract PDFs from subtask metadata
                        pdfs.extend(
                            self._extract_pdfs_from_subtask(prev_subtask)
                        )

        # Also check direct dependencies
        if current_subtask_id in self.state_manager.subtasks:
            current_subtask = self.state_manager.subtasks[current_subtask_id]
            for dep_id in current_subtask.dependencies:
                if dep_id in self.state_manager.subtasks:
                    dep_subtask = self.state_manager.subtasks[dep_id]
                    if dep_subtask.status == 'completed':
                        pdfs.extend(
                            self._extract_pdfs_from_subtask(dep_subtask)
                        )

        # Remove duplicates (by filename)
        seen_filenames = set()
        unique_pdfs = []
        for pdf in pdfs:
            if pdf.filename not in seen_filenames:
                seen_filenames.add(pdf.filename)
                unique_pdfs.append(pdf)

        self.logger.info(
            f'Found {len(unique_pdfs)} unique PDF(s) from previous subtasks'
        )
        return unique_pdfs

    def _extract_pdfs_from_subtask(
        self, subtask: Subtask
    ) -> List[Attachment]:
        """
        Extract PDF attachments from a subtask's metadata.

        Args:
            subtask: Subtask to extract PDFs from.

        Returns:
            List of PDF attachments.
        """
        pdfs = []

        # Check if subtask metadata contains PDF information
        metadata = subtask.metadata or {}

        # Check for downloaded_files in search_metadata
        search_metadata = metadata.get('search_metadata', {})
        downloaded_files = search_metadata.get('downloaded_files', [])

        for file_data in downloaded_files:
            if file_data.get('type') == 'pdf':
                # Try to reconstruct attachment from metadata
                # Note: This is a fallback - ideally PDFs should be stored as attachments
                url = file_data.get('url', '')
                if url:
                    self.logger.debug(
                        f'Found PDF reference in subtask {subtask.id}: {url}'
                    )
                    # We can't reconstruct the full attachment from metadata alone,
                    # but we can log it for reference
                    # The actual PDF should be in the attachments list passed to execute_subtask

        # Check for PDF data in metadata
        pdf_data = metadata.get('pdf_data', {})
        if pdf_data:
            self.logger.debug(
                f'Found PDF data in subtask {subtask.id} metadata'
            )

        return pdfs

    def merge_pdfs_into_attachments(
        self,
        attachments: List[Attachment],
        current_subtask_id: str,
    ) -> List[Attachment]:
        """
        Merge PDFs from previous subtasks into current attachments list.

        Args:
            attachments: Current attachments list.
            current_subtask_id: ID of the current subtask.

        Returns:
            Updated attachments list with PDFs from previous subtasks.
        """
        # Get PDFs from previous subtasks
        previous_pdfs = self.get_pdfs_from_previous_subtasks(
            current_subtask_id
        )

        # Merge PDFs into attachments, avoiding duplicates
        existing_filenames = {att.filename for att in attachments}
        existing_urls = set()
        for att in attachments:
            if hasattr(att, 'metadata') and att.metadata:
                url = att.metadata.get('source_url')
                if url:
                    existing_urls.add(url)

        for pdf in previous_pdfs:
            # Check by filename
            if pdf.filename not in existing_filenames:
                # Check by URL if available
                pdf_url = None
                if hasattr(pdf, 'metadata') and pdf.metadata:
                    pdf_url = pdf.metadata.get('source_url')

                if not pdf_url or pdf_url not in existing_urls:
                    attachments.append(pdf)
                    existing_filenames.add(pdf.filename)
                    if pdf_url:
                        existing_urls.add(pdf_url)
                    self.logger.info(
                        f'Added PDF from previous subtask to context: {pdf.filename}'
                    )

        return attachments

    def store_pdfs_in_subtask_metadata(
        self,
        subtask_id: str,
        pdf_attachments: List[Attachment],
    ) -> None:
        """
        Store PDF attachment references in subtask metadata for later retrieval.

        Args:
            subtask_id: ID of the subtask.
            pdf_attachments: List of PDF attachments to store.
        """
        if subtask_id not in self.state_manager.subtasks:
            return

        subtask = self.state_manager.subtasks[subtask_id]
        if not subtask.metadata:
            subtask.metadata = {}

        # Store PDF metadata (not the full attachment data to avoid duplication)
        pdf_metadata = []
        for pdf in pdf_attachments:
            pdf_info = {
                'filename': pdf.filename,
                'url': (
                    pdf.metadata.get('source_url')
                    if hasattr(pdf, 'metadata') and pdf.metadata
                    else None
                ),
                'content_type': (
                    pdf.content_type
                    if hasattr(pdf, 'content_type')
                    else 'application/pdf'
                ),
                'size': len(pdf.data) if hasattr(pdf, 'data') else 0,
            }
            pdf_metadata.append(pdf_info)

        if 'pdf_attachments' not in subtask.metadata:
            subtask.metadata['pdf_attachments'] = []
        subtask.metadata['pdf_attachments'].extend(pdf_metadata)

        self.logger.debug(
            f'Stored {len(pdf_metadata)} PDF reference(s) in subtask {subtask_id} metadata'
        )



