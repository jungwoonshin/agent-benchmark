"""Image recognition tool for processing images from PDFs and browser navigation."""

import logging
import re
from typing import Any, Dict, List, Optional, Union

from ..models import Attachment
from ..utils.arxiv_utils import (
    extract_arxiv_id_from_text,
    extract_arxiv_id_from_url,
    get_arxiv_metadata,
)


class ImageRecognition:
    """Tool for recognizing and analyzing images from PDFs and browser navigation."""

    def __init__(self, logger: logging.Logger, llm_service=None):
        """
        Initialize ImageRecognition tool.

        Args:
            logger: Logger instance for logging.
            llm_service: Optional LLM service for visual processing.
        """
        self.logger = logger
        self.llm_service = llm_service

    def set_llm_service(self, llm_service):
        """Set the LLM service for visual processing."""
        self.llm_service = llm_service

    def recognize_images_from_pdf(
        self,
        attachment: Attachment,
        options: Optional[Dict[str, Any]] = None,
        problem: Optional[str] = None,
        query_analysis: Optional[Dict[str, Any]] = None,
        skip_image_processing: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Extract and recognize images from a PDF file with smart content filtering.

        Args:
            attachment: PDF attachment to process.
            options: Optional dict with 'page_range' key for specific pages.
            problem: Optional problem description for relevance filtering.
            query_analysis: Optional query analysis for relevance filtering.

        Returns:
            If problem/query_analysis provided: Dictionary with structured data:
            {
                'type': 'pdf',
                'filename': str,
                'sections': [{'title': str, 'page': int, 'content': str}, ...],  # Can be empty list
                'image_analysis': str,
                'full_text': str
            }
            Otherwise: Combined text and image analysis results as string (backward compatibility).
        """
        if options is None:
            options = {}

        try:
            import fitz  # type: ignore # PyMuPDF
        except ImportError:
            error_msg = 'PyMuPDF not available. Install with: uv pip install PyMuPDF'
            self.logger.warning(error_msg)
            return f'Error: {error_msg}'

        self.logger.info(
            f'Extracting and recognizing images from PDF: {attachment.filename}'
        )

        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=attachment.data, filetype='pdf')

            # Determine page range
            page_range = options.get('page_range', None)
            if page_range and isinstance(page_range, list) and len(page_range) == 2:
                start_page = max(0, page_range[0] - 1)  # Convert to 0-based
                end_page = min(len(pdf_document), page_range[1])  # End is inclusive
                pages = range(start_page, end_page)
                self.logger.info(f'Reading PDF pages {page_range[0]}-{page_range[1]}')
            else:
                pages = range(len(pdf_document))
                self.logger.info(f'Reading all {len(pdf_document)} pages of PDF')

            # Step 1: Extract structure (headers/titles) first if problem/query_analysis provided
            relevant_sections = None
            pdf_structure = None
            if problem and query_analysis and self.llm_service:
                self.logger.info(
                    'Extracting PDF structure (headers/titles) for relevance filtering...'
                )
                pdf_structure = self._extract_pdf_structure(pdf_document, pages)
                if pdf_structure:
                    self.logger.info(
                        f'Found {len(pdf_structure)} sections in PDF structure'
                    )
                    # Filter sections by relevance
                    relevant_sections = self._filter_relevant_sections(
                        pdf_structure, problem, query_analysis
                    )
                    if relevant_sections:
                        self.logger.info(
                            f'Identified {len(relevant_sections)} relevant sections out of {len(pdf_structure)} total'
                        )
                    else:
                        self.logger.warning(
                            'No relevant sections found - will skip content extraction (only extract images)'
                        )

            # Step 2: Extract text and images from specified pages
            # Only extract full content from relevant sections if filtering was applied
            text_parts = []
            extracted_images = []
            structured_sections = []  # For structured return format

            for page_num in pages:
                page = pdf_document[page_num]

                # Check if this page should be included (if relevance filtering is active)
                should_extract_text = True
                if relevant_sections is not None:  # None means filtering was attempted
                    # If filtering was attempted but no relevant sections found, still extract text as fallback
                    if len(relevant_sections) == 0:
                        self.logger.debug(
                            f'No relevant sections found in PDF, but extracting full text as fallback for page {page_num + 1}'
                        )
                        # Still extract text, just don't filter by sections
                        should_extract_text = True
                    else:
                        # Check if any relevant section is on this page
                        page_sections = [
                            s
                            for s in relevant_sections
                            if s.get('page', 0) == page_num + 1
                        ]
                        should_extract_text = len(page_sections) > 0
                        if not should_extract_text:
                            self.logger.debug(
                                f'Skipping page {page_num + 1} - no relevant sections on this page'
                            )

                if should_extract_text:
                    # Extract text from page
                    page_text = page.get_text()
                    if page_text.strip():
                        # If relevance filtering was applied (relevant_sections is not None)
                        if relevant_sections is not None:
                            # Extract only from relevant sections
                            page_sections = [
                                s
                                for s in relevant_sections
                                if s.get('page', 0) == page_num + 1
                            ]
                            if page_sections:
                                # Extract text for relevant sections on this page using section_index
                                section_texts = []
                                # Get page blocks for block-based extraction
                                blocks = page.get_text('dict')['blocks']
                                blocks_text_list = []
                                for block in blocks:
                                    block_text = ''
                                    if 'lines' in block:
                                        for line in block['lines']:
                                            if 'spans' in line:
                                                for span in line['spans']:
                                                    block_text += span.get('text', '')
                                    blocks_text_list.append(block_text)

                                for section in page_sections:
                                    section_index = section.get('section_index')
                                    section_title = section.get('title', '')

                                    if section_index is not None and pdf_structure:
                                        # Find the section in pdf_structure by section_index
                                        current_section = None
                                        for s in pdf_structure:
                                            if s.get('section_index') == section_index:
                                                current_section = s
                                                break

                                        if current_section:
                                            # Get block index for current section
                                            start_block_idx = current_section.get(
                                                'block_index', 0
                                            )

                                            # Find next section's block index (section_index + 1)
                                            next_block_idx = len(blocks)
                                            next_section_index = section_index + 1
                                            for s in pdf_structure:
                                                if (
                                                    s.get('section_index')
                                                    == next_section_index
                                                    and s.get('page') == page_num + 1
                                                ):
                                                    next_block_idx = s.get(
                                                        'block_index', len(blocks)
                                                    )
                                                    break

                                            # Extract text from blocks between start and next
                                            section_blocks_text = blocks_text_list[
                                                start_block_idx:next_block_idx
                                            ]
                                            section_text = '\n'.join(
                                                section_blocks_text
                                            ).strip()

                                            if section_text:
                                                section_texts.append(
                                                    f'[Section: {section_title}]\n{section_text}'
                                                )
                                                # Add to structured sections
                                                structured_sections.append(
                                                    {
                                                        'title': section_title,
                                                        'page': page_num + 1,
                                                        'content': section_text,
                                                        'section_index': section_index,
                                                    }
                                                )
                                            else:
                                                # Fallback: try string matching if block extraction yielded no text
                                                self.logger.debug(
                                                    f'Block-based extraction failed for section {section_index}, trying string matching'
                                                )
                                                if section_title in page_text:
                                                    start_idx = page_text.find(
                                                        section_title
                                                    )
                                                    if start_idx >= 0:
                                                        # Find next section on same page
                                                        remaining_text = page_text[
                                                            start_idx:
                                                        ]
                                                        next_section_idx = len(
                                                            remaining_text
                                                        )
                                                        for s in pdf_structure:
                                                            if (
                                                                s.get('page')
                                                                == page_num + 1
                                                                and s.get(
                                                                    'section_index', -1
                                                                )
                                                                > section_index
                                                            ):
                                                                other_title = s.get(
                                                                    'title', ''
                                                                )
                                                                if (
                                                                    other_title
                                                                    in remaining_text
                                                                ):
                                                                    other_idx = remaining_text.find(
                                                                        other_title
                                                                    )
                                                                    if (
                                                                        other_idx > 0
                                                                        and other_idx
                                                                        < next_section_idx
                                                                    ):
                                                                        next_section_idx = other_idx
                                                        section_text = remaining_text[
                                                            :next_section_idx
                                                        ].strip()
                                                        if section_text:
                                                            section_texts.append(
                                                                f'[Section: {section_title}]\n{section_text}'
                                                            )
                                                            # Add to structured sections
                                                            structured_sections.append(
                                                                {
                                                                    'title': section_title,
                                                                    'page': page_num
                                                                    + 1,
                                                                    'content': section_text,
                                                                    'section_index': section_index,
                                                                }
                                                            )

                                if section_texts:
                                    text_parts.append(
                                        f'[Page {page_num + 1}]\n'
                                        + '\n\n'.join(section_texts)
                                    )
                                else:
                                    # Fallback to full page text if section extraction failed
                                    self.logger.debug(
                                        f'Section extraction failed for page {page_num + 1}, using full page text as fallback'
                                    )
                                    if page_text.strip():
                                        text_parts.append(
                                            f'[Page {page_num + 1}]\n{page_text}'
                                        )
                            else:
                                # No relevant sections on this page, but still extract text as fallback
                                if page_text.strip():
                                    text_parts.append(
                                        f'[Page {page_num + 1}]\n{page_text}'
                                    )
                        else:
                            # No relevance filtering applied (problem/query_analysis not provided)
                            # Extract full page text for backward compatibility
                            if page_text.strip():
                                text_parts.append(f'[Page {page_num + 1}]\n{page_text}')

                # Always extract images from all pages (images might be relevant even if text isn't)
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image['image']
                        image_ext = base_image['ext']

                        # Store image with metadata
                        extracted_images.append(
                            {
                                'data': image_bytes,
                                'page': page_num + 1,
                                'index': img_index,
                                'ext': image_ext,
                            }
                        )
                        self.logger.debug(
                            f'Extracted image {img_index + 1} from page {page_num + 1} '
                            f'({len(image_bytes)} bytes, format: {image_ext})'
                        )
                    except Exception as e:
                        self.logger.warning(
                            f'Failed to extract image {img_index} from page {page_num + 1}: {e}'
                        )

            pdf_document.close()

            # Combine text parts
            combined_text = (
                '\n\n'.join(text_parts) if text_parts else '[No text content found]'
            )

            # Build context that includes relevant section titles if available
            context_for_images = combined_text
            if relevant_sections is not None and len(relevant_sections) > 0:
                relevant_titles_info = (
                    '\n\nRelevant Sections Identified:\n'
                    + '\n'.join(
                        f'- Page {s.get("page", "?")}: {s.get("title", "")}'
                        for s in relevant_sections
                    )
                )
                context_for_images = (
                    relevant_titles_info + '\n\n' + combined_text
                    if combined_text != '[No text content found]'
                    else relevant_titles_info
                )
                self.logger.info(
                    f'Including {len(relevant_sections)} relevant section titles in image analysis context'
                )

            # Process images with visual LLM if available and not skipping
            image_analysis = ''
            if not skip_image_processing:
                if extracted_images and self.llm_service:
                    self.logger.info(
                        f'Found {len(extracted_images)} image(s) in PDF. Processing with visual LLM...'
                    )
                    image_analysis = self._process_images_with_visual_llm(
                        extracted_images,
                        task_description=problem
                        or 'Analyze the images from this PDF and extract key information.',
                        context_text=context_for_images,
                        source_type='PDF',
                        source_name=attachment.filename,
                    )
                elif extracted_images:
                    # If no visual LLM available, just note that images were found
                    image_info = ', '.join(
                        f'page {img["page"]} (image {img["index"] + 1})'
                        for img in extracted_images
                    )
                    image_analysis = (
                        f'\n\n[Note: {len(extracted_images)} image(s) found in PDF ({image_info}), '
                        f'but visual LLM not available for analysis]'
                    )
            else:
                # Skip image processing but note that images were found
                if extracted_images:
                    self.logger.debug(
                        f'Skipping image processing: {len(extracted_images)} image(s) found in PDF'
                    )

            # Combine results
            result = combined_text
            if image_analysis and not skip_image_processing:
                result += '\n\n'
                result += 'IMAGE ANALYSIS (from visual LLM):\n'
                result += '\n'
                result += image_analysis

            self.logger.info(
                f'Successfully processed PDF {attachment.filename}. '
                f'Text length: {len(combined_text)}, Images: {len(extracted_images)}'
            )

            # Return structured data if problem/query_analysis provided
            # (sections can be empty - that's okay, we still want type='pdf' for metadata extraction)
            if problem and query_analysis:
                return {
                    'type': 'pdf',
                    'filename': attachment.filename,
                    'sections': structured_sections,  # Can be empty list
                    'image_analysis': image_analysis
                    if not skip_image_processing
                    else '',
                    'full_text': result,  # Include full combined text as fallback
                    'extracted_images': extracted_images
                    if skip_image_processing
                    else None,  # Store images for later processing
                }

            # Backward compatibility: return string when problem/query_analysis not provided
            return result

        except Exception as e:
            self.logger.error(
                f'Failed to process PDF {attachment.filename}: {e}', exc_info=True
            )
            return f'Error: Failed to process PDF {attachment.filename}: {str(e)}'

    def process_pdf_images_after_relevance(
        self,
        attachment: Attachment,
        extracted_images: List[Dict[str, Any]],
        problem: Optional[str] = None,
        context_text: str = '',
    ) -> str:
        """
        Process images from a PDF after relevance has been determined.

        Args:
            attachment: PDF attachment.
            extracted_images: List of extracted images from previous processing.
            problem: Optional problem description for image analysis.
            context_text: Optional context text to help with image analysis.

        Returns:
            Image analysis string.
        """
        if not extracted_images:
            return ''

        if not self.llm_service:
            image_info = ', '.join(
                f'page {img["page"]} (image {img["index"] + 1})'
                for img in extracted_images
            )
            return (
                f'[Note: {len(extracted_images)} image(s) found in PDF ({image_info}), '
                f'but visual LLM not available for analysis]'
            )

        self.logger.info(
            f'Processing {len(extracted_images)} image(s) from PDF {attachment.filename} with visual LLM...'
        )

        return self._process_images_with_visual_llm(
            extracted_images,
            task_description=problem
            or 'Analyze the images from this PDF and extract key information.',
            context_text=context_text,
            source_type='PDF',
            source_name=attachment.filename,
        )

    def recognize_images_from_browser(
        self,
        screenshot_data: bytes,
        context: Optional[Dict[str, Any]] = None,
        task_description: Optional[str] = None,
    ) -> str:
        """
        Recognize and analyze images from browser navigation (screenshots).
        Args:
            screenshot_data: Screenshot image data as bytes.
            context: Optional context dictionary with page information.
            task_description: Optional task description for analysis.
        Returns:
            Analysis result from visual LLM.
        """
        if not screenshot_data:
            self.logger.warning('[Visual LLM Analysis] No screenshot data provided')
            return 'Error: No screenshot data provided'

        if not self.llm_service:
            self.logger.warning(
                '[Visual LLM Analysis] Visual LLM not available for screenshot analysis'
            )
            return '[Visual LLM not available for screenshot analysis]'

        source_name = context.get('url', 'unknown URL') if context else 'unknown URL'
        self.logger.info(
            f'[Visual LLM Analysis] Processing browser screenshot from {source_name} '
            f'({len(screenshot_data)} bytes)'
        )

        context_str = ''
        if context:
            context_items = []
            for key, value in context.items():
                if key in ('screenshot', 'image') or 'image' in key.lower():
                    continue
                context_items.append(f'- {key}: {str(value)[:500]}')
            if context_items:
                context_str = '\n'.join(context_items)
                self.logger.debug(
                    f'[Visual LLM Analysis] Context provided: {len(context_str)} chars'
                )

        final_task_description = (
            task_description
            or 'Analyze this webpage screenshot and determine the next action.'
        )

        return self._process_images_with_visual_llm(
            images=[screenshot_data],
            task_description=final_task_description,
            context_text=context_str,
            source_type='browser',
            source_name=source_name,
        )

    def recognize_images(
        self,
        images: List[bytes],
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        source_type: str = 'general',
        source_name: Optional[str] = None,
    ) -> str:
        """
        Recognize and analyze a list of images using visual LLM.

        Args:
            images: List of image data as bytes.
            task_description: Description of what to analyze in the images.
            context: Optional context dictionary.
            source_type: Type of source (e.g., 'PDF', 'browser', 'general').
            source_name: Optional name of the source.

        Returns:
            Analysis result from visual LLM.
        """
        if not images:
            self.logger.warning('[Visual LLM Analysis] No images provided')
            return '[No images provided]'

        if not self.llm_service:
            self.logger.warning(
                f'[Visual LLM Analysis] Visual LLM not available for {source_type} ({source_name or "unknown"})'
            )
            return '[Visual LLM not available]'

        source_name = source_name or 'unknown'
        self.logger.info(
            f'[Visual LLM Analysis] Processing {len(images)} image(s) from {source_type} ({source_name})'
        )

        return self._process_images_with_visual_llm(
            images,
            task_description=task_description,
            context_text=str(context) if context else '',
            source_type=source_type,
            source_name=source_name,
        )

    def _process_images_with_visual_llm(
        self,
        images: List[Any],
        task_description: str,
        context_text: str = '',
        source_type: str = 'general',
        source_name: str = 'unknown',
    ) -> str:
        """
        Process images with visual LLM.
        Args:
            images: List of images (can be dicts with 'data' key or bytes).
            task_description: Description of what to analyze in the images.
            context_text: Text context for analysis.
            source_type: Type of source.
            source_name: Name of the source.
        Returns:
            Analysis result.
        """
        if not self.llm_service:
            self.logger.warning(
                f'Visual LLM not available for {source_type} ({source_name})'
            )
            return '[Visual LLM not available]'

        try:
            # Log visual LLM analysis start with details
            self.logger.info(
                f'[Visual LLM Analysis] Starting analysis for {source_type} ({source_name})'
            )
            self.logger.info(
                f'[Visual LLM Analysis] Task: {task_description[:200]}{"..." if len(task_description) > 200 else ""}'
            )
            self.logger.info(
                f'[Visual LLM Analysis] Image count: {len(images)}, Context length: {len(context_text)} chars'
            )

            # Build content list with text and images
            content_items = []

            # System prompt is now universal and direct
            system_prompt = """You are an expert at analyzing images to extract relevant information.
Focus on answering the user's request based on the visual content.
Be concise and directly address the user's query.
If the image does not contain relevant information, state that clearly.
Do not describe the image unless asked to."""

            # User prompt is now a clear instruction
            text_prompt = f"""Task: {task_description}
Source: {source_type} ({source_name})"""

            if context_text:
                text_prompt += f'\n\nRelevant Text Context:\n{context_text[:2000]}'  # Limit context length

            text_prompt += '\n\nPlease analyze the following image(s) and provide a direct answer to the task.'

            content_items.append({'type': 'text', 'text': text_prompt})

            # Add all images and log image details
            image_sizes = []
            valid_image_count = 0
            for i, img in enumerate(images):
                try:
                    # Handle both dict format (from PDF) and bytes format
                    if isinstance(img, dict):
                        image_data = img['data']
                        image_format = img.get('ext', 'auto')
                        page_info = (
                            f' (page {img.get("page", "?")})' if 'page' in img else ''
                        )
                    else:
                        image_data = img
                        image_format = 'auto'
                        page_info = ''

                    image_size = len(image_data)
                    image_sizes.append(image_size)
                    image_content = self.llm_service.create_image_content(
                        image_data, image_format=image_format
                    )
                    content_items.append(image_content)
                    valid_image_count += 1
                    self.logger.debug(
                        f'[Visual LLM Analysis] Added image {i + 1}{page_info} ({image_size} bytes, format: {image_format})'
                    )
                except Exception as e:
                    self.logger.warning(
                        f'[Visual LLM Analysis] Failed to encode image {i + 1}: {e}'
                    )

            if len(content_items) == 1:  # Only text, no valid images added
                self.logger.error(
                    '[Visual LLM Analysis] No valid images could be added to request'
                )
                return '[Failed to process images with visual LLM - no valid images]'

            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': content_items},
            ]

            total_image_size = sum(image_sizes)
            self.logger.info(
                f'[Visual LLM Analysis] Request prepared: {valid_image_count} valid image(s), '
                f'total size: {total_image_size} bytes, prompt length: {len(text_prompt)} chars'
            )

            # Use visual LLM to analyze images
            temperature = 0.1  # Lower temperature for more focused analysis
            max_tokens = 8192  # Increased from 4000 to reduce truncation risk
            self.logger.info(
                f'[Visual LLM Analysis] Calling visual LLM with temperature={temperature}, max_tokens={max_tokens}'
            )

            analysis = self.llm_service.call_with_images(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self.logger.info(
                f'[Visual LLM Analysis] Analysis completed: {len(analysis)} chars returned'
            )
            self.logger.info(
                f'[Visual LLM Analysis] Analysis preview: {analysis}{"..." if len(analysis) > 200 else ""}'
            )
            return analysis

        except Exception as e:
            self.logger.error(
                f'[Visual LLM Analysis] Failed to process images from {source_type} ({source_name}): {e}',
                exc_info=True,
            )
            return f'[Error processing images with visual LLM: {str(e)}]'

    def _extract_pdf_structure(
        self, pdf_document: Any, pages: range
    ) -> List[Dict[str, Any]]:
        """
        Extract headers and titles from PDF structure.

        Args:
            pdf_document: PyMuPDF document object.
            pages: Range of pages to process.

        Returns:
            List of sections with title, page, and level information.
        """
        sections = []
        try:
            # Try to extract structure using PyMuPDF's structure
            for page_num in pages:
                page = pdf_document[page_num]
                # Get page text blocks
                blocks = page.get_text('dict')['blocks']

                # Extract potential headers (larger text, bold, at start of blocks)
                for block_idx, block in enumerate(blocks):
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                for span in line['spans']:
                                    text = span.get('text', '').strip()
                                    if not text:
                                        continue

                                    font_size = span.get('size', 0)
                                    flags = span.get('flags', 0)
                                    is_bold = flags & 16  # Bit 4 indicates bold

                                    # Heuristic: Headers are typically:
                                    # - Larger font size (>= 12pt)
                                    # - Bold
                                    # - Short text (typically < 200 chars)
                                    # - At start of line
                                    # - May be all caps or title case
                                    if (
                                        font_size >= 12
                                        and is_bold
                                        and len(text) < 200
                                        and len(text.split()) <= 15
                                    ):
                                        sections.append(
                                            {
                                                'title': text,
                                                'page': page_num + 1,
                                                'level': 1 if font_size >= 16 else 2,
                                                'font_size': font_size,
                                                'block_index': block_idx,
                                            }
                                        )

            # Also try to extract from table of contents if available
            toc = pdf_document.get_toc()
            if toc:
                for item in toc:
                    level, title, page_num_toc = item
                    if page_num_toc - 1 in pages:  # Convert to 0-based
                        # Try to find matching block index on the target page
                        toc_page = pdf_document[page_num_toc - 1]
                        toc_blocks = toc_page.get_text('dict')['blocks']
                        block_index = None
                        # Search for the title in blocks to get approximate position
                        for block_idx, block in enumerate(toc_blocks):
                            # Extract text from this block
                            block_text = ''
                            if 'lines' in block:
                                for line in block['lines']:
                                    if 'spans' in line:
                                        for span in line['spans']:
                                            block_text += span.get('text', '')
                            if (
                                title.lower() in block_text.lower()[:200]
                            ):  # Check first 200 chars
                                block_index = block_idx
                                break
                        sections.append(
                            {
                                'title': title,
                                'page': page_num_toc,
                                'level': level,
                                'font_size': 14,  # Default for TOC entries
                                'block_index': block_index
                                if block_index is not None
                                else 0,
                            }
                        )

            # Remove duplicates (same title on same page)
            seen = set()
            unique_sections = []
            for section in sections:
                key = (section['title'], section['page'])
                if key not in seen:
                    seen.add(key)
                    unique_sections.append(section)

            # Sort sections by page number, then by block_index within page
            unique_sections.sort(key=lambda s: (s['page'], s.get('block_index', 0)))

            # Assign 0-based sequential section_index to each section
            for idx, section in enumerate(unique_sections):
                section['section_index'] = idx

            self.logger.debug(
                f'Extracted {len(unique_sections)} unique sections from PDF structure with indices 0-{len(unique_sections) - 1}'
            )
            return unique_sections

        except Exception as e:
            self.logger.warning(f'Failed to extract PDF structure: {e}')
            return []

    def _filter_relevant_sections(
        self,
        sections: List[Dict[str, Any]],
        problem: str,
        query_analysis: Dict[str, Any],
        subtask_description: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter PDF sections by relevance to problem and query analysis using LLM.

        Args:
            sections: List of section dictionaries with title, page, level.
            problem: Problem description.
            query_analysis: Query analysis dictionary.
            subtask_description: Optional subtask description to extract step number.

        Returns:
            List of relevant sections.
        """
        if not self.llm_service or not sections:
            return sections

        try:
            # Build prompt for relevance checking
            sections_text = '\n'.join(
                [
                    f'- Page {s["page"]}, Level {s["level"]}: {s["title"]}'
                    for s in sections
                ]
            )

            # Extract step number from subtask_description if provided
            step_num = None
            if subtask_description:
                step_match = re.search(
                    r'step[_\s]*(\d+)', subtask_description, re.IGNORECASE
                )
                if step_match:
                    try:
                        step_num = int(step_match.group(1))
                    except (ValueError, IndexError):
                        pass

            # Filter explicit_requirements by step number if step number is available
            explicit_requirements = query_analysis.get('explicit_requirements', [])
            if step_num is not None and explicit_requirements:
                # Filter to only requirements matching this step number
                filtered_requirements = []
                for req in explicit_requirements:
                    # Check if requirement starts with "Step N:" where N matches step_num
                    step_match = re.search(r'Step\s+(\d+):', req, re.IGNORECASE)
                    if step_match:
                        req_step_num = int(step_match.group(1))
                        if req_step_num == step_num:
                            # Remove the "Step N:" prefix for cleaner display
                            req_text = re.sub(
                                r'Step\s+\d+:\s*', '', req, flags=re.IGNORECASE
                            ).strip()
                            filtered_requirements.append(req_text)
                    else:
                        # If no step prefix, include it (backward compatibility)
                        filtered_requirements.append(req)
                explicit_requirements = filtered_requirements

            system_prompt = """You are an expert at analyzing document structure and determining relevance.
Your task is to identify which sections of a PDF document are relevant to answering a specific question.

Consider:
1. The explicit requirements mentioned in the question
2. The context and domain of the question
3. Whether the section title suggests it contains relevant information

Return a JSON object with a "relevant_titles" key containing an array of relevant section titles."""

            # Build user prompt without problem, only include explicit_requirements if they match step
            requirements_text = ''
            if explicit_requirements:
                requirements_text = (
                    f'\nExplicit Requirements: {", ".join(explicit_requirements)}'
                )

            user_prompt = f"""PDF Section Titles:
{sections_text}{requirements_text}

Identify which section titles are relevant based on the explicit requirements.
Return a JSON object with a "relevant_titles" key containing an array of relevant section titles (as strings).
Example: {{"relevant_titles": ["Introduction", "Methodology", "Results"]}}"""

            response = self.llm_service.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
                response_format={'type': 'json_object'},
            )

            # Parse response
            import json

            from ..utils.json_utils import extract_json_from_text

            json_text = extract_json_from_text(response)
            response_data = json.loads(json_text)

            # Extract relevant titles (handle different response formats)
            relevant_titles = set()
            if isinstance(response_data, dict):
                # Look for common keys
                if 'relevant_titles' in response_data:
                    relevant_titles = set(response_data['relevant_titles'])
                elif 'relevant_sections' in response_data:
                    relevant_titles = set(response_data['relevant_sections'])
                elif 'titles' in response_data:
                    relevant_titles = set(response_data['titles'])
                elif 'sections' in response_data:
                    relevant_titles = set(response_data['sections'])
                else:
                    # Try to find any array value
                    for value in response_data.values():
                        if isinstance(value, list):
                            relevant_titles = set(value)
                            break
            elif isinstance(response_data, list):
                relevant_titles = set(response_data)

            # Normalize titles (trim whitespace, case-insensitive matching)
            relevant_titles_normalized = {t.strip().lower() for t in relevant_titles}

            # Filter sections by relevant titles (case-insensitive matching)
            # Preserve section_index from original sections
            relevant_sections = [
                s
                for s in sections
                if s['title'].strip().lower() in relevant_titles_normalized
            ]

            self.logger.info(
                f'LLM identified {len(relevant_titles)} relevant titles out of {len(sections)} total sections'
            )
            if relevant_sections:
                indices = [s.get('section_index', '?') for s in relevant_sections]
                self.logger.debug(f'Relevant section indices: {indices}')

            # Return only relevant sections (no fallback to all sections)
            return relevant_sections

        except Exception as e:
            self.logger.warning(
                f'Failed to filter sections by relevance: {e}. Will skip content extraction.'
            )
            return []  # Return empty list if filtering fails - no fallback to all sections

    def extract_arxiv_metadata_from_pdf(
        self, attachment: Attachment
    ) -> Optional[Dict[str, Any]]:
        """
        Extract arXiv metadata (submission date, paper ID) using arXiv API.
        Extracts paper ID from URL or PDF content, then fetches metadata from arXiv API.

        Args:
            attachment: PDF attachment to extract metadata from.

        Returns:
            Dictionary with keys:
            - paper_id: arXiv paper ID (extracted from URL if available)
            - submission_date: Original submission date (YYYY-MM-DD format if found)
            - submission_date_text: Raw submission date text from PDF
            - submission_month: Month of submission (YYYY-MM format if found)
            - confidence: Confidence score (0.0-1.0)
            None if extraction fails.
        """
        paper_id = None

        # Strategy 1: Extract paper ID from URL (most reliable)
        if hasattr(attachment, 'metadata') and attachment.metadata:
            source_url = attachment.metadata.get('source_url', '')
            if source_url:
                paper_id = extract_arxiv_id_from_url(source_url)
                if paper_id:
                    self.logger.debug(f'Extracted arXiv ID from URL: {paper_id}')

        # Strategy 2: Extract paper ID from PDF text if not found in URL
        if not paper_id:
            try:
                import fitz  # type: ignore # PyMuPDF
            except ImportError:
                fitz = None

            if fitz:
                try:
                    pdf_document = fitz.open(stream=attachment.data, filetype='pdf')
                    if len(pdf_document) > 0:
                        # Extract text from first page (where arXiv ID is typically found)
                        first_page_text = pdf_document[0].get_text()
                        paper_id = extract_arxiv_id_from_text(first_page_text)
                        pdf_document.close()
                        if paper_id:
                            self.logger.debug(
                                f'Extracted arXiv ID from PDF text: {paper_id}'
                            )
                except Exception as e:
                    self.logger.debug(f'Failed to extract arXiv ID from PDF text: {e}')

        # If we have a paper_id, fetch metadata from arXiv API
        if paper_id:
            try:
                metadata = get_arxiv_metadata(paper_id, self.logger)
                if metadata:
                    # Return in the expected format (only include expected keys)
                    result = {
                        'paper_id': metadata.get('paper_id'),
                        'submission_date': metadata.get('submission_date'),
                        'submission_date_text': metadata.get('submission_date_text'),
                        'submission_month': metadata.get('submission_month'),
                        'confidence': metadata.get('confidence', 1.0),
                    }

                    self.logger.info(
                        f'Extracted arXiv metadata from API - Date: {result["submission_date"]}, '
                        f'Paper ID: {result["paper_id"]}, Confidence: {result["confidence"]:.2f}'
                    )
                    return result
                else:
                    self.logger.warning(
                        f'arXiv API returned no metadata for paper ID: {paper_id}'
                    )
            except Exception as e:
                self.logger.warning(
                    f'Failed to fetch metadata from arXiv API for {paper_id}: {e}',
                    exc_info=True,
                )

            # Fallback: return paper_id if we found one, even if API call failed
            self.logger.info(
                f'Found arXiv paper ID {paper_id} but could not fetch metadata from API'
            )
            return {
                'paper_id': paper_id,
                'submission_date': None,
                'submission_date_text': None,
                'submission_month': None,
                'confidence': 0.3,  # Low confidence - only paper ID, no date
            }

        # No paper ID found
        self.logger.debug('No arXiv paper ID found in URL or PDF content')
        return None

    def extract_arxiv_metadata_from_pdf_no_llm(
        self, attachment: Attachment
    ) -> Optional[Dict[str, Any]]:
        """
        Extract arXiv metadata without using LLM - uses PDF metadata and regex patterns.

        This method extracts submission dates using:
        1. PDF document metadata (creation date)
        2. Regex pattern matching on first page text
        3. Header/footer text extraction

        Args:
            attachment: PDF attachment to extract metadata from.

        Returns:
            Dictionary with keys:
            - paper_id: arXiv paper ID (extracted from URL if available)
            - submission_date: Original submission date (YYYY-MM-DD format if found)
            - submission_date_text: Raw submission date text from PDF
            - submission_month: Month of submission (YYYY-MM format if found)
            - confidence: Confidence score (0.0-1.0)
            None if extraction fails.
        """
        try:
            import re

            import fitz  # type: ignore # PyMuPDF
        except ImportError:
            self.logger.warning('PyMuPDF not available for arXiv metadata extraction')
            return None

        try:
            # Extract paper ID from URL if available
            paper_id = None
            if hasattr(attachment, 'metadata') and attachment.metadata:
                source_url = attachment.metadata.get('source_url', '')
                if source_url:
                    # Extract arXiv paper ID from URL
                    arxiv_match = re.search(
                        r'arxiv\.org/(?:abs|pdf)/([\d.]+)', source_url, re.IGNORECASE
                    )
                    if arxiv_match:
                        paper_id = arxiv_match.group(1)
                        # Remove version suffix if present
                        paper_id = re.sub(r'v\d+$', '', paper_id)

            # Open PDF and extract metadata
            pdf_document = fitz.open(stream=attachment.data, filetype='pdf')
            if len(pdf_document) == 0:
                pdf_document.close()
                return {'paper_id': paper_id} if paper_id else None

            # Get PDF document metadata (title, author, creation date, etc.)
            pdf_metadata = pdf_document.metadata
            creation_date = pdf_metadata.get('creationDate', '')

            # Extract first page text (where submission info is typically found)
            first_page = pdf_document[0]
            first_page_text = first_page.get_text().strip()

            # Also try to get text from header/footer areas
            blocks = first_page.get_text('dict')['blocks']
            header_footer_text = ''
            page_height = first_page.rect.height

            for block in blocks:
                if 'bbox' in block and 'lines' in block:
                    bbox = block['bbox']
                    # Check if block is in header (top 15%) or footer (bottom 15%)
                    if bbox[1] < page_height * 0.15 or bbox[3] > page_height * 0.85:
                        for line in block.get('lines', []):
                            if 'spans' in line:
                                for span in line.get('spans', []):
                                    header_footer_text += span.get('text', '') + ' '

            # Combine first page text with header/footer text
            combined_text = (
                first_page_text + '\n\n' + header_footer_text.strip()
            ).strip()

            pdf_document.close()

            if not combined_text:
                self.logger.warning('No text found in first page of PDF')
                return {'paper_id': paper_id} if paper_id else None

            # Use regex patterns to find submission dates
            submission_date = None
            submission_date_text = None
            submission_month = None

            # Pattern 1: "Submitted on [date]" or "Originally submitted [date]"
            # Match various date formats
            date_patterns = [
                # YYYY-MM-DD format
                r'(?:submitted|originally submitted|submission date)[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                # Month DD, YYYY format
                r'(?:submitted|originally submitted|submission date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                # DD Month YYYY format
                r'(?:submitted|originally submitted|submission date)[:\s]+(\d{1,2}\s+[A-Za-z]+\s+\d{4})',
                # MM/DD/YYYY format
                r'(?:submitted|originally submitted|submission date)[:\s]+(\d{1,2}/\d{1,2}/\d{4})',
                # Just look for dates near "submitted" keywords
                r'submitted[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
                r'submitted[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            ]

            for pattern in date_patterns:
                match = re.search(pattern, combined_text, re.IGNORECASE)
                if match:
                    date_str = match.group(1).strip()
                    submission_date_text = match.group(0).strip()
                    # Try to parse the date
                    submission_date = self._parse_date_string(date_str)
                    if submission_date:
                        submission_month = submission_date[:7]  # YYYY-MM
                        break

            # If no submission date found, try to extract from PDF creation date
            if not submission_date and creation_date:
                # PDF dates are in format: D:YYYYMMDDHHmmSSOHH'mm'
                pdf_date_match = re.search(r'D:(\d{4})(\d{2})(\d{2})', creation_date)
                if pdf_date_match:
                    year, month, day = pdf_date_match.groups()
                    submission_date = f'{year}-{month}-{day}'
                    submission_month = f'{year}-{month}'
                    submission_date_text = f'PDF creation date: {creation_date}'

            result = {
                'paper_id': paper_id,
                'submission_date': submission_date,
                'submission_date_text': submission_date_text,
                'submission_month': submission_month,
                'confidence': 0.8 if submission_date else 0.0,
            }

            if result['submission_date']:
                self.logger.info(
                    f'Extracted arXiv submission date (regex): {result["submission_date"]}'
                )
            else:
                self.logger.debug('No submission date found in PDF content (regex)')

            return result

        except Exception as e:
            self.logger.warning(
                f'Failed to extract arXiv metadata from PDF (regex): {e}', exc_info=True
            )
            return None

    def _parse_date_string(self, date_str: str) -> Optional[str]:
        """
        Parse various date string formats into YYYY-MM-DD format.

        Args:
            date_str: Date string in various formats.

        Returns:
            Date in YYYY-MM-DD format or None if parsing fails.
        """
        import re
        from datetime import datetime

        # Try different date formats
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%B %d, %Y',  # June 15, 2022
            '%B %d %Y',  # June 15 2022
            '%d %B %Y',  # 15 June 2022
            '%m/%d/%Y',  # 06/15/2022
            '%d/%m/%Y',  # 15/06/2022
        ]

        # Clean up the date string
        date_str = date_str.strip()

        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # Try to extract year-month-day from numeric formats
        numeric_match = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
        if numeric_match:
            year, month, day = numeric_match.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        return None
