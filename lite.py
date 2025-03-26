import base64
import requests
from typing import Union, List, Dict, Optional, Any, cast
from crewai import LLM
import json
import litellm
from litellm import Choices
from litellm.types.utils import ModelResponse

class LLMWithMultimodalSupport(LLM):
    def __init__(self, image_path: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_path = image_path

    def _prepare_multimodal_messages(
        self, 
        messages: Union[str, List[Dict[str, Union[str, List[Dict[str, Any]]]]]],
        images: Optional[List[Union[str, bytes, Dict[str, str]]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages with multimodal support for Gemini models.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        prepared_messages = messages.copy()
        
        if images:
            image_parts = []
            for image in images:
                if isinstance(image, str):
                    if image.startswith(('http://', 'https://')):
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
                    else:
                        with open(image, 'rb') as img_file:
                            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                            image_parts.append({
                                "type": "image",
                                "image": base64_image
                            })
                elif isinstance(image, bytes):
                    base64_image = base64.b64encode(image).decode('utf-8')
                    image_parts.append({
                        "type": "image",
                        "image": base64_image
                    })
                elif isinstance(image, dict):
                    image_parts.append(image)
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")

            if prepared_messages[-1]['role'] == 'user':
                if isinstance(prepared_messages[-1]['content'], list):
                    prepared_messages[-1]['content'].extend(image_parts)
                else:
                    prepared_messages[-1]['content'] = [
                        {"type": "text", "text": prepared_messages[-1]['content']}
                    ] + image_parts
        
        return prepared_messages

    def _handle_non_streaming_response(
        self,
        params: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Handle a non-streaming response from the LLM."""
        image_input = [self.image_path] if self.image_path else None
        params["messages"] = self._prepare_multimodal_messages(
            params["messages"], 
            images=image_input
        )
        
        response = litellm.completion(**params)
        response_message = cast(Choices, cast(ModelResponse, response).choices)[0].message
        return response_message.content or ""

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes, Dict[str, str]]]] = None,
    ) -> Union[str, Any]:
        """
        Extended call method to support multimodal inputs for Gemini models.
        """
        multimodal_models = ['gemini', 'google']
        if images and any(model in self.model.lower() for model in multimodal_models):
            messages = self._prepare_multimodal_messages(messages, images)
        
        return super().call(
            messages=messages, 
            tools=tools, 
            callbacks=callbacks, 
            available_functions=available_functions
        )
