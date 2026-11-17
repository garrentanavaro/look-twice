# Demo page

Page used for the user study, served by the backend at http://localhost:8000/demo/.

## Clips

Put the clips in sets:

```
demo/clips/A/clip01.mp4 ... clip05.mp4
demo/clips/B/clip01.mp4 ... clip05.mp4
```

Use neutral file names, since they are visible in the browser. Keep the answer key somewhere outside the `demo/` folder. Video files are not committed to the repository.

## Session

1. Open `/demo/?pid=P01&set=A&phase=unassisted` with the extension turned off.
2. The participant watches each clip and answers real or deepfake, plus how sure they are (1 to 5).
3. Switch to the other set with `phase=assisted` and turn the extension on.
4. At the end, click "Download all answers (CSV)".

Answers are stored in the browser, so run all sessions on the same laptop.
