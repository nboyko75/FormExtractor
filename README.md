# FormExtractor

FormExtractor is an application for recognizing text in scanned forms from PDF and image files and extracting it into parameter name–value pairs. It is designed to return the extracted data as plain text or JSON for review, storage, or further processing.

## Project status

This repository is in its initial setup stage. The application code, installation steps, and usage commands are not yet available. The workflow and examples below describe the intended behavior.

## Planned capabilities

- Read scanned forms from PDF and image files.
- Use optical character recognition (OCR) to recognize text in scans.
- Identify form field labels as parameter names and associate them with their values.
- Return the extracted parameter names and values in plain text or JSON format.

Specific image formats and OCR languages will be documented when support is implemented.

## Intended workflow

1. Provide a PDF or image containing a scanned form.
2. Recognize the form's text using OCR.
3. Extract parameter names and their corresponding values.
4. Return the results in the selected output format: text or JSON.

For example, a form containing fields labeled `Full Name`, `Date of Birth`, and `Application Number` would produce the name–value pairs shown below.

## Output examples

These examples illustrate the proposed output structure; the final interface may change during development.

### Plain text

```text
Full Name: Alex Morgan
Date of Birth: 1990-05-21
Application Number: APP-00123
```

### JSON

```json
{
  "parameters": [
    {
      "param_name": "Full Name",
      "param_value": "Alex Morgan"
    },
    {
      "param_name": "Date of Birth",
      "param_value": "1990-05-21"
    },
    {
      "param_name": "Application Number",
      "param_value": "APP-00123"
    }
  ]
}
```

In this proposed format, each parameter has a `param_name` containing the field label and a `param_value` containing its recognized value. Values are represented as strings to preserve text such as identifiers with leading zeros.

## Recognition quality

OCR accuracy depends on scan quality, text readability, and form layout. Clear, correctly oriented scans are preferable. Extracted names and values should be reviewed when accuracy is important.
