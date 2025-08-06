import os
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1



project_id = os.getenv("PROJECT_ID")
processor_location = os.getenv("PROCESSOR_LOCATION")
processor_id = os.getenv("PROCESSOR_ID")

opts = ClientOptions(api_endpoint=f"{processor_location}-documentai.googleapis.com")
client = documentai_v1.DocumentProcessorServiceClient(client_options=opts)
full_processor_name = client.processor_path(project_id, processor_location, processor_id)
request = documentai_v1.GetProcessorRequest(name=full_processor_name)
processor = client.get_processor(request=request)

def ocr(image_content):
    raw_document = documentai_v1.RawDocument(
        content=image_content,
        mime_type="application/pdf",
    )
    request = documentai_v1.ProcessRequest(name=processor.name, raw_document=raw_document)
    result = client.process_document(request=request)
    document = result.document
    return document.text