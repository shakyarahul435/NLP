import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .rag_service import get_evaluation_summary, get_example_before_after, get_pipeline


@csrf_exempt
def health(request):
    return JsonResponse({"status": "ok"})


@csrf_exempt
def evaluation(request):
    if request.method != "GET":
        return JsonResponse({"error": "Use GET"}, status=405)

    return JsonResponse(
        {
            "evaluation_table": get_evaluation_summary(),
            "example_before_after": get_example_before_after(),
        }
    )


@csrf_exempt
def ask(request):
    if request.method != "POST":
        return JsonResponse({"error": "Use POST"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    question = (payload.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Question is required"}, status=400)

    pipeline = get_pipeline()
    answer, chunks = pipeline.answer(question, mode="contextual", top_k=3)
    top = chunks[0]

    return JsonResponse(
        {
            "question": question,
            "answer": answer,
            "source": {
                "document": top.source,
                "chunk_id": top.chunk_id,
                "before": top.text,
                "after": top.contextual_text,
            },
            "evaluation_table": get_evaluation_summary(),
            "example_before_after": get_example_before_after(),
        }
    )
