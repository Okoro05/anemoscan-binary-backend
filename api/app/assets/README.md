# Report branding assets

- `anemoscan-logo.svg` is the official vector wordmark used at the top of generated PDF scan reports.
- Keep the SVG viewBox and intrinsic proportions intact. The report renderer scales it to fit the A4 page width.
- If replacing the logo, retain the filename or update the `BRANDING_LOGO` path in `api/app/admin_routes.py`.
- The API has a PDF regression test that verifies the logo can be rendered into a valid PDF.
