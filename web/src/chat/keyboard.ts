/**
 * Whether a key press belongs to an input method rather than to the page.
 *
 * With a Chinese (or Japanese, Korean) IME, the Enter that commits pinyin to
 * text also reaches the page as a keydown. Chrome marks it `isComposing`;
 * Safari ends the composition first and sends that keydown with
 * `isComposing` false, so its keyCode 229 ("IME processing") is the only
 * sign. Checking both keeps "confirm the pinyin" from sending the message.
 */
export function isImeKeyEvent(event: { nativeEvent: KeyboardEvent }): boolean {
  return event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229;
}
