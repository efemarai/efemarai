from efemarai.fields import BoundingBox, InstanceMask, Polygon


def show_sample(
    original_datapoint,
    generated_datapoint,
    model_output,
    baseline_loss,
    sample_loss,
    delta_score,
):
    import cv2
    import numpy as np

    image = np.array(generated_datapoint.image.data[:, :, ::-1])
    cv2.putText(
        image,
        f"{delta_score:.4f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    boxes = [field for field in model_output.outputs if isinstance(field, BoundingBox)]
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)

    polygons = [
        field
        for field in model_output.outputs
        if isinstance(field, (Polygon, InstanceMask))
    ]
    for poly in polygons:
        vertices = None

        if isinstance(poly, Polygon):
            vertices = poly.vertices

        if isinstance(poly, InstanceMask):
            vertices = poly.to_polygon().vertices

        if vertices:
            for vertice in vertices:
                cv2.polylines(
                    image,
                    np.array(vertice)[None, :, :].astype(int),
                    True,
                    (255, 255, 255),
                    1,
                )

    cv2.imshow(f"Generated Image", image)
    cv2.waitKey(1)
