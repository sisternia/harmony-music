import { Stack, usePathname } from "expo-router";
import { StatusBar } from "expo-status-bar";
import { useLayoutEffect } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { colors } from "@/theme/colors";

export default function RootLayout() {
  const path = usePathname();

  useLayoutEffect(() => {
    if (Platform.OS !== "web") return;
    const section = path === "/" ? "New transcription"
      : path === "/piano" ? "Piano"
      : path.startsWith("/songscription") ? "Songscription"
      : path === "/projects" ? "Projects"
      : path === "/collections" ? "Collections"
      : path.startsWith("/p/") ? "Studio"
      : path === "/auth" ? "Login"
      : "Harmony Auralis";
    document.title = `${section} - Harmony Auralis`;
  }, [path]);

  return (
    <View style={styles.shell}>
      <StatusBar style="dark" />
      <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: colors.background } }} />
    </View>
  );
}

const styles = StyleSheet.create({
  shell: { flex: 1, backgroundColor: colors.background },
});
